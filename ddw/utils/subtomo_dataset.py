import os

import time
import torch
from scipy import spatial
from torch.utils.data import Dataset

from .fourier import apply_fourier_mask_to_tomo
from .missing_wedge import (get_missing_wedge_mask,
                            get_rotated_missing_wedge_mask)
from .rotation import rotate_vol_around_axis

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



class SubtomoDataset(Dataset):
    """
    A torch dataset which produces the input-target sub-tomogram pairs used for model fitting. The directory 'subtomo_dir' must have the same structure as the output of the 'ddw prepare-data' command.
    """

    def __init__(
        self,
        subtomo_dir,
        crop_subtomos_to_size,
        mw_angle=None,
        rotate_subtomos=True,
        deterministic_rotations=False,
    ):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        self.crop_subtomos_to_size = crop_subtomos_to_size
        self.mw_angle = mw_angle
        # if subtomo_dir contains a 'ctf' subdirectory, i.e. a file '{index}.pt' with
        # the same shape as 'subtomo0/{index}.pt' and 'subtomo1/{index}.pt' for every
        # index, a per-subtomo 3D-CTF/mask (values in [0, 1], where 1 means a Fourier
        # component is fully trusted and 0 means it is missing) is used instead of a
        # binary wedge mask generated on the fly from mw_angle
        self.use_ctf = os.path.isdir(f"{subtomo_dir}/ctf")
        if not self.use_ctf and mw_angle is None:
            raise ValueError(
                f"mw_angle must be provided because '{subtomo_dir}' does not contain "
                "a 'ctf' subdirectory."
            )
        self.rotate_subtomos = rotate_subtomos
        self.deterministic_rotations = deterministic_rotations

    @property
    def rotate_subtomos(self):
        return self._rotate_subtomos

    @rotate_subtomos.setter
    def rotate_subtomos(self, rotate_subtomos):
        if not isinstance(rotate_subtomos, bool):
            raise ValueError("rotate_subtomos must be a boolean")
        self._rotate_subtomos = rotate_subtomos

    def _sample_rot_axis_and_angle(self, index):
        seed = BASE_SEED + index if self.deterministic_rotations else None
        rotvec = torch.from_numpy(
            spatial.transform.Rotation.random(random_state=seed).as_rotvec()
        )
        rot_axis = rotvec / rotvec.norm()
        rot_angle = torch.rad2deg(rotvec.norm())
        return rot_axis, rot_angle

    def __len__(self):
        return len(os.listdir(f"{self.subtomo_dir}/subtomo0"))

    def __getitem__(self, index):
        # load subtomos
        subtomo0_file = f"{self.subtomo_dir}/subtomo0/{index}.pt"
        subtomo0 = safe_load(subtomo0_file)
        subtomo1_file = f"{self.subtomo_dir}/subtomo1/{index}.pt"
        subtomo1 = safe_load(subtomo1_file)
        if self.use_ctf:
            ctf = safe_load(f"{self.subtomo_dir}/ctf/{index}.pt")
        # rotate subtomos
        if self.rotate_subtomos == True:
            rot_axis, rot_angle = self._sample_rot_axis_and_angle(index)
            subtomo0 = rotate_vol_around_axis(
                subtomo0,
                rot_angle=rot_angle,
                rot_axis=rot_axis,
                output_shape=3 * [self.crop_subtomos_to_size],
            )
            subtomo1 = rotate_vol_around_axis(
                subtomo1,
                rot_angle=rot_angle,
                rot_axis=rot_axis,
                output_shape=3 * [self.crop_subtomos_to_size],
            )
            # add missing wedge/CTF
            if self.use_ctf:
                # mw_mask plays the role of a "canonical", un-rotated wedge/CTF that
                # is imposed on the already-rotated subtomo0 as additional corruption
                # for self-supervised training (see get_missing_wedge_mask below for
                # the binary-wedge equivalent). Since we only have one real CTF per
                # subtomo pair, we reuse it here too, only center-cropped (rot_angle=0).
                mw_mask = rotate_vol_around_axis(
                    ctf,
                    rot_angle=0,
                    rot_axis=rot_axis,
                    output_shape=3 * [self.crop_subtomos_to_size],
                )
                # rot_mw_mask tracks the real, underlying CTF geometry as the
                # sub-tomogram is rotated for augmentation
                rot_mw_mask = rotate_vol_around_axis(
                    ctf,
                    rot_angle=rot_angle,
                    rot_axis=rot_axis,
                    output_shape=3 * [self.crop_subtomos_to_size],
                )
            else:
                mw_mask = get_missing_wedge_mask(
                    grid_size=3 * [self.crop_subtomos_to_size],
                    mw_angle=self.mw_angle,
                    device=subtomo0.device,
                )
                rot_mw_mask = get_rotated_missing_wedge_mask(
                    grid_size=3 * [self.crop_subtomos_to_size],
                    mw_angle=self.mw_angle,
                    rot_axis=rot_axis,
                    rot_angle=rot_angle,
                    device=subtomo0.device,
                )
        else:
            if self.use_ctf:
                mw_mask = ctf
            else:
                mw_mask = get_missing_wedge_mask(
                    grid_size=subtomo0.shape,
                    mw_angle=self.mw_angle,
                    device=subtomo0.device,
                )
            rot_mw_mask = mw_mask
            rot_angle, rot_axis = 0, torch.tensor([1.0, 0.0, 0.0])

        model_input = apply_fourier_mask_to_tomo(subtomo0, mw_mask)
        item = {
            "model_input": model_input,
            "model_target": subtomo1,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "rot_angle": rot_angle,
            "rot_axis": rot_axis,
        }
        return item


# %%
