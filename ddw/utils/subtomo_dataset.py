import os

import time
import torch
from scipy import spatial
from torch.utils.data import Dataset

from .fourier import apply_fourier_mask_to_tomo, rfft_mask_to_full_mask
from .missing_wedge import (get_missing_wedge_mask,
                            get_rotated_missing_wedge_mask)
from .rotation import rotate_fourier_mask_around_axis, rotate_vol_around_axis

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
        # if subtomo_dir contains 'ctf' and 'ctf_crop' subdirectories, a per-subtomo
        # 3D-CTF/mask (values in [0, 1], where 1 means a Fourier component is fully
        # trusted and 0 means it is missing) is used instead of a binary wedge mask
        # generated on the fly from mw_angle. 'ctf/{index}.pt' must match
        # 'subtomo0/{index}.pt'/'subtomo1/{index}.pt' (the on-disk/native size);
        # 'ctf_crop/{index}.pt' must match crop_subtomos_to_size. Since a CTF (unlike
        # the binary wedge) has genuine radial/magnitude frequency dependence, it
        # cannot be correctly resized in Fourier space - so ctf_crop must be
        # independently, correctly computed at its own smaller resolution, not
        # derived by cropping/resizing 'ctf'. Each tensor may be given either as a
        # full (N, N, N) array (DC at the center, fftshift convention - matching
        # get_missing_wedge_mask) or as a real-valued rfftn-convention array of shape
        # (N, N, N//2+1) with DC at index [0,0,0] (unshifted); the latter is expanded
        # to the former automatically on load (see rfft_mask_to_full_mask).
        self.use_ctf = os.path.isdir(f"{subtomo_dir}/ctf")
        if self.use_ctf and not os.path.isdir(f"{subtomo_dir}/ctf_crop"):
            raise ValueError(
                f"'{subtomo_dir}' contains a 'ctf' subdirectory but no 'ctf_crop' "
                "subdirectory; both are required together."
            )
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
            ctf = rfft_mask_to_full_mask(safe_load(f"{self.subtomo_dir}/ctf/{index}.pt"))
        # rotate subtomos
        if self.rotate_subtomos == True:
            rot_axis, rot_angle = self._sample_rot_axis_and_angle(index)
            subtomo1 = rotate_vol_around_axis(
                subtomo1,
                rot_angle=rot_angle,
                rot_axis=rot_axis,
                output_shape=3 * [self.crop_subtomos_to_size],
            )
            if self.use_ctf:
                # A CTF (unlike the binary wedge) has genuine radial/magnitude
                # frequency dependence, so it cannot be correctly resized in Fourier
                # space. We therefore rotate subtomo0 and ctf at their shared native,
                # on-disk size (no resizing), apply the CTF there, and only crop the
                # *resulting real-space volume* down to crop_subtomos_to_size
                # afterward - real-space cropping is a valid spatial-windowing
                # operation, unlike resizing a Fourier-domain array.
                subtomo0_native = rotate_vol_around_axis(
                    subtomo0, rot_angle=rot_angle, rot_axis=rot_axis, output_shape=None
                )
                # ctf is used un-rotated here: it plays the role of a "canonical"
                # corruption imposed on the already-rotated subtomo0, exactly as
                # get_missing_wedge_mask (always evaluated in the canonical
                # orientation) does for the binary-wedge equivalent below.
                model_input_native = apply_fourier_mask_to_tomo(subtomo0_native, ctf)
                model_input = rotate_vol_around_axis(
                    model_input_native,
                    rot_angle=0,
                    rot_axis=rot_axis,
                    output_shape=3 * [self.crop_subtomos_to_size],
                )
                # mw_mask/rot_mw_mask are only used for the loss weighting, where
                # model_output/target live at crop_subtomos_to_size. ctf_crop is a
                # *separately, correctly computed* CTF at that resolution (not a
                # resize of ctf), so rotating it in place is valid (no resizing
                # happens: it's already at crop_subtomos_to_size).
                ctf_crop = rfft_mask_to_full_mask(safe_load(f"{self.subtomo_dir}/ctf_crop/{index}.pt"))
                mw_mask = ctf_crop
                rot_mw_mask = rotate_fourier_mask_around_axis(
                    ctf_crop,
                    rot_angle=rot_angle,
                    rot_axis=rot_axis,
                    output_shape=3 * [self.crop_subtomos_to_size],
                )
            else:
                subtomo0 = rotate_vol_around_axis(
                    subtomo0,
                    rot_angle=rot_angle,
                    rot_axis=rot_axis,
                    output_shape=3 * [self.crop_subtomos_to_size],
                )
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
                model_input = apply_fourier_mask_to_tomo(subtomo0, mw_mask)
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

        item = {
            "model_target": subtomo1,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "rot_angle": rot_angle,
            "rot_axis": rot_axis,
        }
        if self.rotate_subtomos == True:
            # model_input is not computed when rotate_subtomos is False: no consumer
            # needs it there (update_subtomo_missing_wedges re-masks subtomo0_pure
            # itself), so the key is omitted entirely rather than set to a placeholder
            # - any accidental access fails loudly with a KeyError instead of
            # silently getting a stale/masked-but-unrotated volume.
            item["model_input"] = model_input
        return item


# %%
