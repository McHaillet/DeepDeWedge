"""
End-to-end tests for 'ddw fit-model' with a per-subtomo 3D-CTF subtomo_dir, and
regression checks that the legacy mw_angle path is unaffected. These call fit_model()
directly (not through the CLI). Most of these actually run the U-Net, which requires a
GPU: fit_model always builds its pytorch_lightning Trainer with accelerator="gpu".
"""
import pytest
import torch

from ddw.fit_model import fit_model

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fit_model always builds its Trainer with accelerator='gpu'",
)


def _fit_kwargs(subtomo_dir, logdir, crop_size, num_downsample_layers=3, mw_angle=None):
    kwargs = dict(
        unet_params_dict={"chans": 4, "num_downsample_layers": num_downsample_layers, "drop_prob": 0.0},
        adam_params_dict={"lr": 1e-3},
        num_epochs=2,
        batch_size=2,
        subtomo_size=crop_size,
        gpu=[0],
        num_workers=2,
        subtomo_dir=str(subtomo_dir),
        logdir=str(logdir),
        logger="csv",
        check_val_every_n_epochs=1,
        # force the periodic subtomo-refresh step (the one with the size-divisibility
        # requirement) to fire on every epoch
        update_subtomo_missing_wedges_every_n_epochs=1,
        save_model_every_n_epochs=100,
        save_n_models_with_lowest_fitting_loss=0,
        save_n_models_with_lowest_val_loss=0,
        seed=0,
    )
    if mw_angle is not None:
        kwargs["mw_angle"] = mw_angle
    return kwargs


def test_fit_model_requires_mw_angle_without_ctf(make_subtomo_dir, tmp_path):
    # raises before the Trainer/GPU is ever touched, so no GPU needed here
    root = make_subtomo_dir(native_size=32, crop_size=24, n_fitting=4, n_val=0, with_ctf=False)
    with pytest.raises(ValueError, match="mw_angle"):
        fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


@requires_gpu
def test_fit_model_with_ctf_completes(make_subtomo_dir, tmp_path):
    root = make_subtomo_dir(native_size=32, crop_size=24, n_fitting=6, n_val=2, with_ctf=True)
    fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


@requires_gpu
def test_fit_model_ctf_raises_on_indivisible_native_size(make_subtomo_dir, tmp_path):
    # on-disk/native size (30) not divisible by 2**num_downsample_layers (8): the CTF
    # path must raise rather than silently pad, since a CTF can't be correctly resized
    root = make_subtomo_dir(native_size=30, crop_size=24, n_fitting=6, n_val=2, with_ctf=True)
    with pytest.raises(ValueError, match="divisible"):
        fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


@requires_gpu
def test_fit_model_legacy_pads_for_indivisible_native_size(make_subtomo_dir, tmp_path):
    # regression check: the legacy mw_angle path must still complete via padding for a
    # non-divisible native size, since prepare_data.py's sqrt(2)-margin enlargement
    # doesn't reliably land on a size divisible by 2**num_downsample_layers
    root = make_subtomo_dir(native_size=30, crop_size=24, n_fitting=6, n_val=2, with_ctf=False)
    fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24, mw_angle=50))


@requires_gpu
def test_subtomo0_pure_created_and_never_modified(make_subtomo_dir, tmp_path):
    """
    update_subtomo_missing_wedges must snapshot subtomo0 into subtomo0_pure on its
    first run and never touch that snapshot again, no matter how many update cycles
    run afterward - it's the permanent anchor that keeps a continuous CTF mask's
    real-data contribution from eroding towards the model's own prediction over
    repeated updates (see the design discussion this fixes: naively re-blending
    "currently on disk" with "model prediction" every cycle is a contraction that
    converges to 100% model-predicted content even at near-fully-trusted frequencies).
    """
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=6, n_val=2, with_ctf=True)
    subtomo0_dir = root / "fitting_subtomos" / "subtomo0"
    pure_dir = root / "fitting_subtomos" / "subtomo0_pure"

    original = {f.name: torch.load(f) for f in subtomo0_dir.glob("*.pt")}
    assert not pure_dir.exists()

    # 2 epochs, update_subtomo_missing_wedges_every_n_epochs=1 -> two update cycles
    fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=crop))

    assert pure_dir.exists()
    for name, original_tensor in original.items():
        assert torch.equal(torch.load(pure_dir / name), original_tensor), (
            f"subtomo0_pure/{name} must stay byte-identical to the original extraction"
        )
    # subtomo0 itself is the one that's actually supposed to change
    changed = any(
        not torch.equal(torch.load(subtomo0_dir / name), original_tensor)
        for name, original_tensor in original.items()
    )
    assert changed, "subtomo0 should have been refreshed by the update step"

    # val_subtomos gets its own independent snapshot too
    assert (root / "val_subtomos" / "subtomo0_pure").exists()
