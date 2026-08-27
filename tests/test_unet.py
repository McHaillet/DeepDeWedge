"""
Tests for LitUnet3D's rotation helpers (ddw.utils.unet): _sample_rotations/_rotate_batch.
Pure CPU/torch - no GPU needed (only fit_model's Trainer requires one).
"""
import random

import torch

from ddw.utils.unet import LitUnet3D


def _make_lit_unet():
    return LitUnet3D(
        unet_params=dict(chans=2, num_downsample_layers=1),
        adam_params=dict(lr=1e-3),
        subtomo_size=8,
    )


def test_rotate_batch_round_trips_with_shared_rot_mats_when_deterministic():
    torch.manual_seed(0)
    lit_unet = _make_lit_unet()
    vol = torch.randn(3, 6, 6, 6)
    indices = [0, 5, 12]

    rot_mats = lit_unet._sample_rotations(indices, deterministic=True)
    rotated = lit_unet._rotate_batch(vol, rot_mats)
    back = lit_unet._rotate_batch(rotated, rot_mats, inverse=True)
    assert torch.allclose(back, vol)


def test_rotate_batch_round_trips_with_shared_rot_mats_when_non_deterministic():
    """
    Regression test: sample_grid_rotation(index, deterministic=False) draws from the shared
    global 'random' state, so two independent calls with the same 'index' are not guaranteed
    to agree - _sample_rotations must be called once per step and its result reused for both
    the forward and inverse _rotate_batch call, not re-derived from 'indices' each time.
    Advance the global random state between the two calls to make sure this is actually
    exercised.
    """
    torch.manual_seed(0)
    random.seed(1)
    lit_unet = _make_lit_unet()
    vol = torch.randn(3, 6, 6, 6)
    indices = [0, 5, 12]

    rot_mats = lit_unet._sample_rotations(indices, deterministic=False)
    rotated = lit_unet._rotate_batch(vol, rot_mats)
    random.random()  # perturb the shared global random state before the "inverse" call
    back = lit_unet._rotate_batch(rotated, rot_mats, inverse=True)
    assert torch.allclose(back, vol)
