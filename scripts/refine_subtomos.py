"""
Run a fitted DeepDeWedge U-Net directly on a pre-existing pool of even/odd
subvolumes, bypassing subvolume extraction entirely.

Unlike `ddw refine-tomogram`, which crops subvolumes from full tomograms with
a sliding window (see ddw/utils/subtomos.py:extract_subtomos) before running
the model, this script assumes the subvolumes have *already* been cropped by
some other pipeline and just feeds them through the model as-is. This is
useful for isolating whether a custom subvolume-preparation step - rather
than the model itself or refine-tomogram's own cropping - is responsible for
an issue.

For every filename common to --even-dir and --odd-dir (each holding
torch.save'd (D, D, D) tensors, e.g. "0.pt", "1.pt", ...), this script:
  1. loads the even and odd subvolume,
  2. runs each independently through the model (the model normalizes inputs
     internally using its own checkpointed normalization_loc/
     normalization_scale - subvolumes are NOT rescaled beforehand, so this
     also surfaces whether your subvolumes' intensity distribution is
     compatible with the checkpoint),
  3. averages the two model outputs (mirrors ddw/refine_tomogram.py, which
     averages the refined tomo0 and tomo1),
  4. saves the result to --output-dir/{filename}.

Filenames present in only one of the two directories are skipped with a
warning.

Usage:
    python refine_subtomos.py --even-dir subtomos/subtomo0 --odd-dir subtomos/subtomo1 \\
        --model-checkpoint logs/version_0/checkpoints/.../epoch=99.ckpt \\
        --output-dir refined_subtomos --device cuda:0
"""

import argparse
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from ddw.fit_model import LitUnet3D


class PairedSubtomoDataset(Dataset):
    """
    Loads filename-matched even/odd subvolume pairs from two directories of
    torch.save'd (D, D, D) tensors.
    """

    def __init__(self, even_dir: Path, odd_dir: Path):
        even_files = {p.name for p in even_dir.glob("*.pt")}
        odd_files = {p.name for p in odd_dir.glob("*.pt")}
        only_even = even_files - odd_files
        only_odd = odd_files - even_files
        if only_even:
            print(
                f"WARNING: {len(only_even)} file(s) in '{even_dir}' have no "
                f"counterpart in '{odd_dir}', skipping: {sorted(only_even)}"
            )
        if only_odd:
            print(
                f"WARNING: {len(only_odd)} file(s) in '{odd_dir}' have no "
                f"counterpart in '{even_dir}', skipping: {sorted(only_odd)}"
            )
        self.filenames = sorted(even_files & odd_files)
        if not self.filenames:
            raise ValueError(
                f"No matching filenames found between '{even_dir}' and '{odd_dir}'."
            )
        self.even_dir = even_dir
        self.odd_dir = odd_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        even = torch.load(self.even_dir / filename).float()
        odd = torch.load(self.odd_dir / filename).float()
        return even, odd, filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--even-dir", type=Path, required=True,
        help="Directory of even subvolumes (torch.save'd (D, D, D) tensors).",
    )
    parser.add_argument(
        "--odd-dir", type=Path, required=True,
        help="Directory of odd subvolumes (torch.save'd (D, D, D) tensors), "
        "with filenames matching --even-dir.",
    )
    parser.add_argument(
        "--model-checkpoint", type=Path, required=True,
        help="Path to a DeepDeWedge model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to save the refined subvolumes.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for model inference."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of CPU workers for data loading.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Device to run the model on, e.g. "cpu", "cuda", or "cuda:0".',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = LitUnet3D.load_from_checkpoint(
        args.model_checkpoint, map_location=args.device
    ).to(args.device).eval()

    dataset = PairedSubtomoDataset(args.even_dir, args.odd_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(
        f"Refining {len(dataset)} even/odd subvolume pair(s) from "
        f"'{args.even_dir}' and '{args.odd_dir}'."
    )
    with torch.no_grad():
        for even_batch, odd_batch, filenames in tqdm.tqdm(loader, desc="Refining subvolumes"):
            even_out = model(even_batch.to(args.device)).cpu()
            odd_out = model(odd_batch.to(args.device)).cpu()
            refined_batch = (even_out + odd_out) / 2
            for refined, filename in zip(refined_batch, filenames):
                torch.save(refined.clone(), args.output_dir / filename)

    print(f"Done. Saved {len(dataset)} refined subvolume(s) to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
