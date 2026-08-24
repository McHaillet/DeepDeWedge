"""
Reconstruct even/odd subtomograms on a regular grid of positions for a
single tilt series .xml file.

A grid of box centers is built that evenly covers the tilt series' volume
dimensions at `--box-size` (the native/on-disk box), with boxes overlapping
their neighbors by at least `--overlap` (default 0.1) along each axis. At
every grid position, an even and an odd subtomogram are reconstructed at
`--box-size` from the movies' even/odd frame averages.

Unlike `make_even_odd_subtomos.py`, this script takes a single .xml file
(not a directory of them), does not reconstruct a 3D-CTF, and does not split
the output into fitting/val sets - it just writes every subvolume, in grid
order, to a flat "even"/"odd" directory pair:

    <output_dir>/even/{0,1,...}.pt
    <output_dir>/odd/{0,1,...}.pt

This layout matches what `refine_subtomos.py` expects for --even-dir/--odd-dir.

Reconstruction (subpixel cropping, backprojection) runs on `--device` (default
"cpu"); pass e.g. "cuda" or "cuda:0" to reconstruct on GPU. Saved .pt files are
always moved back to CPU first, so they load fine regardless of device.
`--batch-size` caps how many grid positions are reconstructed in a single
backprojection call, chunking large tomograms to bound peak device memory
(default: no chunking, one call for the whole tilt series).

Usage:
    python make_even_odd_subtomos_single.py /path/to/tilt_series.xml --pixel-size 10.0 \\
        --box-size 136 --output-dir /path/to/output_dir --device cuda --batch-size 32
"""

import argparse
import math
from pathlib import Path

import torch

from warpylib import TiltSeries


def make_grid_positions(volume_dims: torch.Tensor, box_physical: float, overlap: float) -> torch.Tensor:
    """
    Evenly spaced box-center coordinates (Angstrom) that cover volume_dims
    along each axis, with neighboring boxes overlapping by at least `overlap`.
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


def batched_reconstruct(fn, positions: torch.Tensor, batch_size: "int | None") -> torch.Tensor:
    """
    Call fn(positions_chunk) over batch_size-sized chunks of positions (or all
    at once if batch_size is None), moving each chunk's result to CPU right
    away and concatenating into a single CPU tensor. Bounds peak device memory
    for tomograms with many grid positions.
    """
    if batch_size is None:
        return fn(positions).cpu()
    chunks = [fn(positions[i:i + batch_size]).cpu() for i in range(0, positions.shape[0], batch_size)]
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml_file", type=Path, help="Path to a single tilt series .xml file")
    parser.add_argument("--pixel-size", type=float, required=True, help="Reconstruction pixel size in Angstrom")
    parser.add_argument("--box-size", type=int, required=True, help="Subtomogram box size in pixels (must be even)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to create 'even' and 'odd' subdirectories in")
    parser.add_argument("--overlap", type=float, default=0.1, help="Minimum fractional overlap between neighboring grid positions, relative to --box-size (default: 0.1)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to reconstruct on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Max grid positions reconstructed in a single backprojection call; splits large tomograms into chunks to bound memory use (default: no chunking)")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.box_size % 2 != 0:
        raise SystemExit("--box-size must be even")
    if not args.xml_file.is_file():
        raise SystemExit(f"No such file: {args.xml_file}")

    box_physical = args.box_size * args.pixel_size

    ts = TiltSeries(str(args.xml_file)).to(device)
    positions = make_grid_positions(ts.volume_dimensions_physical, box_physical, args.overlap)
    print(f"{args.xml_file.name}: {positions.shape[0]} positions")

    even_dir = args.output_dir / "even"
    odd_dir = args.output_dir / "odd"
    even_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    # load_images always returns CPU tensors (mrcfile reads), so move them onto
    # `device` explicitly; ts itself already lives there via TiltSeries.to(device).
    positions_dev = positions.to(device)

    original_pixel_size = ts.ctf.pixel_size
    _, images_odd, images_even = ts.load_images(
        original_pixel_size=original_pixel_size,
        desired_pixel_size=args.pixel_size,
        load_averages=False,
        load_half_averages=True,
    )
    images_odd = images_odd.to(device)
    images_even = images_even.to(device)

    even_vols = batched_reconstruct(
        lambda p: ts.reconstruct_subvolumes_single(images_even, p, pixel_size=args.pixel_size, size=args.box_size, apply_ctf=True, correct_attenuation=True),
        positions_dev, args.batch_size,
    )
    odd_vols = batched_reconstruct(
        lambda p: ts.reconstruct_subvolumes_single(images_odd, p, pixel_size=args.pixel_size, size=args.box_size, apply_ctf=True, correct_attenuation=True),
        positions_dev, args.batch_size,
    )

    for local_idx in range(positions.shape[0]):
        torch.save(even_vols[local_idx].clone(), even_dir / f"{local_idx}.pt")
        torch.save(odd_vols[local_idx].clone(), odd_dir / f"{local_idx}.pt")

    print(f"Done: {positions.shape[0]} even/odd subvolume pairs saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
