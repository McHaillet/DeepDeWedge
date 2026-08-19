"""
Reconstruct even/odd subtomograms (+ shared 3D-CTF) on a regular grid of
positions for every tilt series .xml in a directory.

For each tilt series, a grid of box centers is built that evenly covers the
tilt series' volume dimensions at `--box-size` (the native/on-disk box),
with boxes overlapping their neighbors by at least `--overlap` (default 0.1)
along each axis. At every grid position, an even and an odd subtomogram are
reconstructed at `--box-size` from the movies' even/odd frame averages, plus
two 3D-CTFs (identical for even and odd, since CTF does not depend on which
half of the frames was used): one at `--box-size` matching the subtomograms
exactly, and one independently reconstructed at the smaller `--crop-box-size`
(the fit-model `subtomo_size`, e.g. after crop_subtomos_to_size) — this is
not a resize/crop of the native CTF, it's its own reconstruction.

All subvolumes from all tilt series are pooled and randomly split into a
fitting and a validation set (`--val-fraction`, default 0.2). Output layout:

    <directory>/subtomos/fitting_subtomos/subtomo0/{0,1,...}.pt    (even, box-size)
    <directory>/subtomos/fitting_subtomos/subtomo1/{0,1,...}.pt    (odd, box-size)
    <directory>/subtomos/fitting_subtomos/ctf/{0,1,...}.pt         (box-size)
    <directory>/subtomos/fitting_subtomos/ctf_crop/{0,1,...}.pt    (crop-box-size)
    <directory>/subtomos/val_subtomos/subtomo0/{0,1,...}.pt        (even, box-size)
    <directory>/subtomos/val_subtomos/subtomo1/{0,1,...}.pt        (odd, box-size)
    <directory>/subtomos/val_subtomos/ctf/{0,1,...}.pt             (box-size)
    <directory>/subtomos/val_subtomos/ctf_crop/{0,1,...}.pt        (crop-box-size)

Reconstruction (subpixel cropping, backprojection) runs on `--device` (default
"cpu"); pass e.g. "cuda" or "cuda:0" to reconstruct on GPU. Saved .pt files are
always moved back to CPU first, so they load fine regardless of device.
`--batch-size` caps how many grid positions are reconstructed in a single
backprojection call, chunking tomograms with many positions to bound peak
device memory (default: no chunking, one call per tomogram).

Usage:
    python make_even_odd_subtomos.py /path/to/warp/dir --pixel-size 10.0 \\
        --box-size 136 --crop-box-size 96 --device cuda --batch-size 32
"""

import argparse
import math
import random
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
    parser.add_argument("directory", type=Path, help="Directory containing tilt series .xml files")
    parser.add_argument("--pixel-size", type=float, required=True, help="Reconstruction pixel size in Angstrom")
    parser.add_argument("--box-size", type=int, required=True, help="Native/on-disk subtomogram + CTF box size in pixels (must be even)")
    parser.add_argument("--crop-box-size", type=int, required=True, help="Smaller box size in pixels for the second, independently reconstructed CTF (must be even)")
    parser.add_argument("--overlap", type=float, default=0.1, help="Minimum fractional overlap between neighboring grid positions, relative to --box-size (default: 0.1)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of subvolumes assigned to the validation set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the fitting/validation split (default: 0)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to reconstruct on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Max grid positions reconstructed in a single backprojection call; splits large tomograms into chunks to bound memory use (default: no chunking)")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.box_size % 2 != 0:
        raise SystemExit("--box-size must be even")
    if args.crop_box_size % 2 != 0:
        raise SystemExit("--crop-box-size must be even")

    directory = args.directory
    xml_paths = sorted(directory.glob("*.xml"))
    if not xml_paths:
        raise SystemExit(f"No .xml files found in {directory}")

    box_physical = args.box_size * args.pixel_size

    # Load metadata and build the position grid for every tilt series up front.
    entries = []  # (xml_path, ts, positions)
    for xml_path in xml_paths:
        ts = TiltSeries(str(xml_path)).to(device)
        positions = make_grid_positions(ts.volume_dimensions_physical, box_physical, args.overlap)
        entries.append((xml_path, ts, positions))

    # Pool every (tilt series, grid position) pair and split randomly.
    tasks = [
        (entry_idx, local_idx)
        for entry_idx, (_, _, positions) in enumerate(entries)
        for local_idx in range(positions.shape[0])
    ]
    random.Random(args.seed).shuffle(tasks)
    n_val = round(len(tasks) * args.val_fraction)
    split_by_task = {}
    for out_idx, task in enumerate(tasks[:n_val]):
        split_by_task[task] = ("val", out_idx)
    for out_idx, task in enumerate(tasks[n_val:]):
        split_by_task[task] = ("fitting", out_idx)

    # Create the output directory layout.
    subtomos_dir = directory / "subtomos"
    output_dirs = {}
    for split in ("fitting", "val"):
        split_dir = subtomos_dir / f"{split}_subtomos"
        output_dirs[split] = {name: split_dir / name for name in ("subtomo0", "subtomo1", "ctf", "ctf_crop")}
        for sub_dir in output_dirs[split].values():
            sub_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct and save.
    for entry_idx, (xml_path, ts, positions) in enumerate(entries):
        print(f"[{entry_idx + 1}/{len(entries)}] {xml_path.name}: {positions.shape[0]} positions")

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
            lambda p: ts.reconstruct_subvolumes_single(images_even, p, pixel_size=args.pixel_size, size=args.box_size),
            positions_dev, args.batch_size,
        )
        odd_vols = batched_reconstruct(
            lambda p: ts.reconstruct_subvolumes_single(images_odd, p, pixel_size=args.pixel_size, size=args.box_size),
            positions_dev, args.batch_size,
        )
        # Shared 3D-CTF: identical for even and odd, so reconstructed once per position.
        # Two independent reconstructions, one per box size (ctf_crop is not a resize of ctf).
        ctf_vols = batched_reconstruct(
            lambda p: ts.reconstruct_subvolume_ctfs_single(p, pixel_size=args.pixel_size, size=args.box_size),
            positions_dev, args.batch_size,
        )
        ctf_crop_vols = batched_reconstruct(
            lambda p: ts.reconstruct_subvolume_ctfs_single(p, pixel_size=args.pixel_size, size=args.crop_box_size),
            positions_dev, args.batch_size,
        )

        for local_idx in range(positions.shape[0]):
            split, out_idx = split_by_task[(entry_idx, local_idx)]
            dirs = output_dirs[split]
            torch.save(even_vols[local_idx].clone(), dirs["subtomo0"] / f"{out_idx}.pt")
            torch.save(odd_vols[local_idx].clone(), dirs["subtomo1"] / f"{out_idx}.pt")
            torch.save(ctf_vols[local_idx].clone(), dirs["ctf"] / f"{out_idx}.pt")
            torch.save(ctf_crop_vols[local_idx].clone(), dirs["ctf_crop"] / f"{out_idx}.pt")

    n_fit = len(tasks) - n_val
    print(f"Done: {n_fit} fitting, {n_val} val subvolumes from {len(entries)} tilt series.")


if __name__ == "__main__":
    main()
