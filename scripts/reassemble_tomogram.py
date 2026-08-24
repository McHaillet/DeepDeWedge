"""
Reassemble a directory of index-matched (0.pt, 1.pt, ...) subvolumes - e.g.
refine_subtomos.py's --output-dir, or the raw "even"/"odd" directories from
make_even_odd_subtomos_single.py - back into a single tomogram.

The grid of box-center positions used to place each subvolume is
*regenerated* from the same tilt series .xml file and the same
--pixel-size/--box-size/--overlap used for extraction (see
make_even_odd_subtomos_single.py) - the grid is deterministic given those
inputs, so no position bookkeeping needs to have been saved during
extraction. --subtomo-dir must contain exactly one file per grid position,
index-matched in the same grid order (0.pt, 1.pt, ...).

Overlapping subvolumes are blended with a linear ramp towards each
subvolume's edges (see ddw.utils.subtomos.reassemble_subtomos). The ramp
width, in voxels, is derived from --overlap and --box-size the same way the
grid spacing itself is: neighboring boxes are spaced box_size * (1 - overlap)
voxels apart, so they overlap by round(box_size * overlap) voxels - this is
not a separate parameter to tune, it must produce the same overlap that was
actually used to build the grid.

Since make_even_odd_subtomos_single.py reconstructs each subvolume at a
sub-voxel-precise physical position (independent per-position backprojection,
not cropped from one pre-existing voxel grid), placement here is necessarily
an approximation: each subvolume is dropped into the output tomogram at its
*nearest* integer voxel corner. Fine for visually inspecting the result, not
a sub-voxel-accurate reconstruction.

Usage:
    python reassemble_tomogram.py /path/to/tilt_series.xml --pixel-size 10.0 \\
        --box-size 136 --overlap 0.1 --subtomo-dir refined_subtomos \\
        --output-file refined_tomogram.mrc
"""

import argparse
import math
from pathlib import Path

import torch

from warpylib import TiltSeries

from ddw.utils.mrctools import save_mrc_data
from ddw.utils.subtomos import reassemble_subtomos


def make_grid_positions(volume_dims: torch.Tensor, box_physical: float, overlap: float) -> torch.Tensor:
    """
    Evenly spaced box-center coordinates (Angstrom) that cover volume_dims
    along each axis, with neighboring boxes overlapping by at least `overlap`.
    Must stay identical to make_even_odd_subtomos_single.py's version, so the
    grid regenerated here matches the one used for extraction.
    """
    step = box_physical * (1.0 - overlap)
    axis_centers = []
    for length in volume_dims.tolist():
        if length <= box_physical:
            centers = torch.tensor([length / 2.0])
        else:
            n = math.ceil((length - box_physical) / step) + 1
            centers = torch.linspace(box_physical / 2.0, length - box_physical / 2.0, n)
        axis_centers.append(centers)
    return torch.cartesian_prod(*axis_centers).reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml_file", type=Path, help="Path to the tilt series .xml file used for extraction")
    parser.add_argument("--pixel-size", type=float, required=True, help="Same --pixel-size used for extraction")
    parser.add_argument("--box-size", type=int, required=True, help="Same --box-size used for extraction")
    parser.add_argument("--overlap", type=float, default=0.1, help="Same --overlap used for extraction (default: 0.1)")
    parser.add_argument("--subtomo-dir", type=Path, required=True, help="Directory of index-matched (0.pt, 1.pt, ...) subvolumes to reassemble")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to save the reassembled tomogram (.mrc)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device used to read the tilt series metadata, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    args = parser.parse_args()

    if not args.xml_file.is_file():
        raise SystemExit(f"No such file: {args.xml_file}")

    ts = TiltSeries(str(args.xml_file)).to(torch.device(args.device))
    box_physical = args.box_size * args.pixel_size
    positions = make_grid_positions(ts.volume_dimensions_physical, box_physical, args.overlap)
    # positions are box *centers*; convert to nearest-voxel start corners, clamped
    # to >= 0 for the edge case where the tilt series volume is smaller than one box
    start_coords = torch.round(positions / args.pixel_size - args.box_size / 2).clamp(min=0).to(torch.int64)
    tomo_shape = torch.round(ts.volume_dimensions_physical / args.pixel_size).to(torch.int64)
    # ts.volume_dimensions_physical (and therefore positions/start_coords/tomo_shape
    # derived from it above) is ordered X,Y,Z, but the subvolume tensors warpylib's
    # reconstruct_subvolumes_single actually returns are axis-ordered Z,Y,X.
    # reassemble_subtomos indexes tensors along their native axes, so start_coords
    # and tomo_shape must be reversed to Z,Y,X to line up with them.
    start_coords = start_coords.flip(-1)
    tomo_shape = tomo_shape.flip(-1)

    n = start_coords.shape[0]
    subtomo_files = [args.subtomo_dir / f"{i}.pt" for i in range(n)]
    missing = [f for f in subtomo_files if not f.is_file()]
    if missing:
        raise SystemExit(
            f"Expected {n} subvolume(s) (one per grid position regenerated from "
            f"'{args.xml_file.name}'), but {len(missing)} are missing from "
            f"'{args.subtomo_dir}', e.g. '{missing[0]}'."
        )
    subtomos = [torch.load(f).float() for f in subtomo_files]
    for subtomo, file in zip(subtomos, subtomo_files):
        if tuple(subtomo.shape) != (args.box_size,) * 3:
            raise SystemExit(
                f"Expected subvolumes of shape {(args.box_size,) * 3}, got "
                f"{tuple(subtomo.shape)} in '{file}'."
            )

    # same overlap, in voxels, that spacing the grid by box_size * (1 - overlap)
    # actually produces between neighboring boxes; None (no ramp, uniform average)
    # if --overlap is 0, since reassemble_subtomos' ramp requires a nonzero width.
    subtomo_overlap = round(args.box_size * args.overlap) or None

    tomo = reassemble_subtomos(
        subtomos=subtomos,
        subtomo_start_coords=start_coords.tolist(),
        subtomo_overlap=subtomo_overlap,
        crop_to_size=tomo_shape.tolist(),
    )

    print(f"Reassembled {n} subvolume(s) into a tomogram of shape {tuple(tomo.shape)}.")
    print(f"Saving to '{args.output_file}'.")
    save_mrc_data(tomo.cpu(), str(args.output_file), save=True)


if __name__ == "__main__":
    main()
