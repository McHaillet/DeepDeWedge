"""
Tests for the per-subtomo 3D-CTF support in SubtomoDataset, as an alternative to the
binary missing-wedge mask generated on the fly from mw_angle. Pure CPU/torch - no GPU
needed.
"""
import pytest
import torch

from ddw.utils.rotation import rotate_vol_around_axis
from ddw.utils.subtomo_dataset import SubtomoDataset


def test_ctf_without_ctf_crop_raises(tmp_path):
    root = tmp_path / "subtomo_dir" / "fitting_subtomos"
    (root / "subtomo0").mkdir(parents=True)
    (root / "subtomo1").mkdir(parents=True)
    (root / "ctf").mkdir(parents=True)
    torch.save(torch.randn(32, 32, 32), root / "subtomo0" / "0.pt")
    torch.save(torch.randn(32, 32, 32), root / "subtomo1" / "0.pt")
    torch.save(torch.rand(32, 32, 32), root / "ctf" / "0.pt")

    with pytest.raises(ValueError, match="ctf_crop"):
        SubtomoDataset(subtomo_dir=str(root), crop_subtomos_to_size=24)


def test_mw_angle_required_without_ctf(make_subtomo_dir):
    root = make_subtomo_dir(native_size=32, crop_size=24, n_fitting=2, n_val=1, with_ctf=False)
    with pytest.raises(ValueError, match="mw_angle"):
        SubtomoDataset(subtomo_dir=str(root / "fitting_subtomos"), crop_subtomos_to_size=24)


def test_legacy_mw_angle_path_still_produces_binary_mask(make_subtomo_dir):
    root = make_subtomo_dir(native_size=32, crop_size=24, n_fitting=2, n_val=1, with_ctf=False)
    ds = SubtomoDataset(
        subtomo_dir=str(root / "fitting_subtomos"), crop_subtomos_to_size=24, mw_angle=50
    )
    assert ds.use_ctf is False
    item = ds[0]
    assert torch.all((item["mw_mask"] == 0) | (item["mw_mask"] == 1))


def test_ctf_item_shapes_and_value_range(make_subtomo_dir):
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=4, n_val=2, with_ctf=True)
    ds = SubtomoDataset(
        subtomo_dir=str(root / "fitting_subtomos"),
        crop_subtomos_to_size=crop,
        rotate_subtomos=True,
        deterministic_rotations=True,
    )
    assert ds.use_ctf is True
    item = ds[0]
    for key in ["model_input", "model_target", "mw_mask", "rot_mw_mask"]:
        assert item[key].shape == (crop, crop, crop)
    assert item["mw_mask"].min() >= -1e-4
    assert item["mw_mask"].max() <= 1 + 1e-4
    # continuous, not the legacy binary wedge
    assert not torch.all((item["mw_mask"] == 0) | (item["mw_mask"] == 1))


def test_ctf_no_rotate_uses_raw_native_ctf(make_subtomo_dir):
    """
    update_subtomo_missing_wedges sets rotate_subtomos=False; in that mode subtomo0
    stays at its native on-disk size, so mw_mask must be the raw 'ctf' tensor
    (unrotated, uncropped) rather than 'ctf_crop'.
    """
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=4, n_val=2, with_ctf=True)
    ds = SubtomoDataset(
        subtomo_dir=str(root / "fitting_subtomos"), crop_subtomos_to_size=crop, rotate_subtomos=False
    )
    item = ds[0]
    assert item["model_input"].shape == (native, native, native)
    assert item["mw_mask"].shape == (native, native, native)
    raw_ctf = torch.load(root / "fitting_subtomos" / "ctf" / "0.pt")
    assert torch.equal(item["mw_mask"], raw_ctf)
    assert torch.equal(item["mw_mask"], item["rot_mw_mask"])


def test_model_input_matches_rotated_subtomo_when_ctf_is_ones(make_subtomo_dir):
    """
    Correctness check for the native-rotate -> native-mask -> real-space-crop
    pipeline: with ctf==1 everywhere, masking is a no-op, so model_input must equal
    subtomo0 rotated and cropped directly to crop_subtomos_to_size (the same
    crop-offset math used for model_target). This guards against model_input and
    model_target silently drifting out of spatial alignment.
    """
    native, crop = 32, 24
    root = make_subtomo_dir(
        native_size=native, crop_size=crop, n_fitting=2, n_val=0, with_ctf=True, ctf_ones=True
    )
    ds = SubtomoDataset(
        subtomo_dir=str(root / "fitting_subtomos"),
        crop_subtomos_to_size=crop,
        rotate_subtomos=True,
        deterministic_rotations=True,
    )
    rot_axis, rot_angle = ds._sample_rot_axis_and_angle(0)
    subtomo0_raw = torch.load(root / "fitting_subtomos" / "subtomo0" / "0.pt")
    expected = rotate_vol_around_axis(
        subtomo0_raw, rot_angle=rot_angle, rot_axis=rot_axis, output_shape=3 * [crop]
    )

    item = ds[0]
    max_err = (item["model_input"] - expected).abs().max().item()
    assert max_err < 1e-3, f"model_input should match rotated+cropped subtomo0, got max_err={max_err}"
