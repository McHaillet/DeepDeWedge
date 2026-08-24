import os

import time
import torch
from scipy import spatial
from torch.utils.data import Dataset

BASE_SEED = 888


def safe_load(file_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return torch.load(file_path)
        # except everything to catch all exceptions
        except Exception as e:
            print(f"Error loading {file_path}")
            if attempt == max_retries - 1:
                raise e  # Reraise if it's the last attempt
            print(f"Error message is: {e}")
            print(f"Retrying in {delay} seconds")
            time.sleep(delay)  # Wait before retrying


def sample_rot_axis_and_angle(index, deterministic):
    """
    Samples a random 3D rotation (axis + angle in degrees). If 'deterministic' is True, the
    rotation only depends on 'index' (via BASE_SEED + index), giving reproducible rotations,
    e.g. for validation.
    """
    seed = BASE_SEED + index if deterministic else None
    rotvec = torch.from_numpy(
        spatial.transform.Rotation.random(random_state=seed).as_rotvec()
    )
    rot_axis = rotvec / rotvec.norm()
    rot_angle = torch.rad2deg(rotvec.norm())
    return rot_axis, rot_angle


class SubtomoDataset(Dataset):
    """
    A torch dataset producing raw (subtomo0, subtomo1, ctf) triples used for model fitting.
    'subtomo_dir' must have 'subtomo0/', 'subtomo1/' and 'ctf/' subdirectories, each holding
    matching '{index}.pt' files.

    'subtomo0'/'subtomo1' are two independent-noise reconstructions of the same tomogram
    region (e.g. from even/odd tilt series frames), at their native, on-disk size - which
    must be larger than the 'subtomo_size' used for model fitting, so that
    LitUnet3D can rotate the model's own estimate and crop down to 'subtomo_size' without
    zero-padding artifacts (see prepare_data's extract_larger_subtomos_for_rotating / the
    box-size vs. crop-box-size split in scripts/make_even_odd_subtomos.py). Both share the
    same physical 'ctf' (values in [0, 1]), since it depends only on the acquisition geometry,
    not on which half of the frames was used.

    'ctf/{index}.pt' must be given at 'subtomo_size' (not the native subtomo0/subtomo1 size):
    a CTF, unlike a binary missing-wedge mask, has genuine radial/magnitude frequency
    dependence and can't be correctly resized in Fourier space, so it must be independently,
    correctly reconstructed at that resolution rather than derived from a native-resolution
    CTF. Each tensor must be real-valued, in rfftn convention: shape (N, N, N//2+1), DC at
    index [0,0,0] (unshifted) - matching what apply_fourier_mask_to_tomo expects. Unlike a
    real-space volume, this mask is never rotated anywhere in this codebase, so there is no
    need to convert it to a full, fftshifted (N, N, N) array (see apply_fourier_mask_to_tomo).
    """

    def __init__(self, subtomo_dir):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        for sub in ["subtomo0", "subtomo1", "ctf"]:
            if not os.path.isdir(f"{subtomo_dir}/{sub}"):
                raise ValueError(f"'{subtomo_dir}' must contain a '{sub}' subdirectory.")

    def __len__(self):
        return len(os.listdir(f"{self.subtomo_dir}/subtomo0"))

    def __getitem__(self, index):
        subtomo0_file = f"{self.subtomo_dir}/subtomo0/{index}.pt"
        subtomo1_file = f"{self.subtomo_dir}/subtomo1/{index}.pt"
        subtomo0 = safe_load(subtomo0_file)
        subtomo1 = safe_load(subtomo1_file)
        ctf = safe_load(f"{self.subtomo_dir}/ctf/{index}.pt")
        return {
            "subtomo0": subtomo0,
            "subtomo1": subtomo1,
            "ctf": ctf,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "index": index,
        }


# %%
