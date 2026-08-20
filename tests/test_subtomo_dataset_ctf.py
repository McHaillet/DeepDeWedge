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
    # model_input is not computed in this path (no consumer needs it; see
    # update_subtomo_missing_wedges, which re-masks subtomo0_pure itself), so the
    # key is omitted entirely rather than set to a placeholder
    assert "model_input" not in item
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


def test_rfftn_shaped_ctf_is_auto_converted(tmp_path):
    """
    ctf/ctf_crop files may be supplied in rfftn convention (half-size last axis, DC
    at [0,0,0], unshifted) instead of the full (N, N, N) fftshifted layout; they must
    be auto-converted on load to give identical results to supplying the pre-converted
    full form directly.
    """
    from ddw.utils.fourier import rfft_mask_to_full_mask

    native, crop = 16, 12
    torch.manual_seed(0)
    # the rfftn-shaped half arrays are the "authoritative" representation here (as a
    # real rfftn-convention CTF file would be); the full/fftshifted equivalent is
    # derived from them via the same conversion SubtomoDataset applies internally, so
    # both directories represent the exact same underlying mask
    ctf_native_half = torch.rand(native, native, native // 2 + 1)
    ctf_crop_half = torch.rand(crop, crop, crop // 2 + 1)
    ctf_native_full = rfft_mask_to_full_mask(ctf_native_half)
    ctf_crop_full = rfft_mask_to_full_mask(ctf_crop_half)

    def build(root, ctf_native, ctf_crop):
        split = root / "fitting_subtomos"
        for sub in ["subtomo0", "subtomo1", "ctf", "ctf_crop"]:
            (split / sub).mkdir(parents=True)
        subtomo0 = torch.randn(native, native, native)
        subtomo1 = torch.randn(native, native, native)
        torch.save(subtomo0, split / "subtomo0" / "0.pt")
        torch.save(subtomo1, split / "subtomo1" / "0.pt")
        torch.save(ctf_native, split / "ctf" / "0.pt")
        torch.save(ctf_crop, split / "ctf_crop" / "0.pt")
        return split

    # dataset A: ctf/ctf_crop given already as full (N, N, N), fftshifted
    root_full = build(tmp_path / "full", ctf_native_full, ctf_crop_full)
    # dataset B: same underlying masks, but given in rfftn convention - SubtomoDataset
    # must convert these to match dataset A exactly
    root_half = build(tmp_path / "half", ctf_native_half, ctf_crop_half)

    ds_full = SubtomoDataset(
        subtomo_dir=str(root_full), crop_subtomos_to_size=crop, deterministic_rotations=True
    )
    ds_half = SubtomoDataset(
        subtomo_dir=str(root_half), crop_subtomos_to_size=crop, deterministic_rotations=True
    )
    # both datasets load different subtomo0/subtomo1 (independent random draws in
    # `build`), so compare the masks directly rather than the full item
    item_full = ds_full[0]
    item_half = ds_half[0]
    assert torch.allclose(item_full["mw_mask"], item_half["mw_mask"], atol=1e-5)
    assert torch.allclose(item_full["rot_mw_mask"], item_half["rot_mw_mask"], atol=1e-5)
