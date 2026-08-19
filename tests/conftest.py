import pytest
import torch


def _build_subtomo_dir(root, native_size, crop_size, n_fitting, n_val, with_ctf, ctf_ones):
    """
    Builds a synthetic subtomo_dir with the layout SubtomoDataset/fit_model expect:
        {root}/{fitting,val}_subtomos/{subtomo0,subtomo1}/{idx}.pt
        {root}/{fitting,val}_subtomos/{ctf,ctf_crop}/{idx}.pt   (if with_ctf)
    'ctf' is at native_size (matching subtomo0/subtomo1); 'ctf_crop' is at crop_size.
    """
    for split, n in [("fitting_subtomos", n_fitting), ("val_subtomos", n_val)]:
        subdirs = ["subtomo0", "subtomo1"] + (["ctf", "ctf_crop"] if with_ctf else [])
        for sub in subdirs:
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for idx in range(n):
            s0 = torch.randn(native_size, native_size, native_size)
            s1 = s0 + 0.1 * torch.randn(native_size, native_size, native_size)
            torch.save(s0, root / split / "subtomo0" / f"{idx}.pt")
            torch.save(s1, root / split / "subtomo1" / f"{idx}.pt")
            if with_ctf:
                if ctf_ones:
                    ctf_native = torch.ones(native_size, native_size, native_size)
                    ctf_crop = torch.ones(crop_size, crop_size, crop_size)
                else:
                    ctf_native = torch.rand(native_size, native_size, native_size).clamp(0, 1)
                    ctf_crop = torch.rand(crop_size, crop_size, crop_size).clamp(0, 1)
                torch.save(ctf_native, root / split / "ctf" / f"{idx}.pt")
                torch.save(ctf_crop, root / split / "ctf_crop" / f"{idx}.pt")


@pytest.fixture
def make_subtomo_dir(tmp_path):
    """
    Factory fixture. Call as:
        make_subtomo_dir(native_size, crop_size, n_fitting, n_val, with_ctf, ctf_ones=False)
    Returns the subtomo_dir root (a Path), containing 'fitting_subtomos'/'val_subtomos'.
    """

    def _make(native_size, crop_size, n_fitting, n_val, with_ctf, ctf_ones=False, name="subtomo_dir"):
        root = tmp_path / name
        _build_subtomo_dir(root, native_size, crop_size, n_fitting, n_val, with_ctf, ctf_ones)
        return root

    return _make
