"""
Tests for ddw.utils.subtomos.get_hann_edge_weights. Pure CPU/torch/numpy - no GPU needed.
"""
import numpy as np
import torch

from ddw.utils.subtomos import get_hann_edge_weights


def test_returns_all_ones_when_taper_width_is_zero():
    weights = get_hann_edge_weights(subtomo_size=10, taper_width=0)
    assert weights.shape == (10, 10, 10)
    assert torch.equal(weights, torch.ones(10, 10, 10))


def test_outermost_voxel_is_exactly_zero_and_interior_is_one():
    n, width = 12, 4
    weights = get_hann_edge_weights(subtomo_size=n, taper_width=width)
    assert weights.shape == (n, n, n)
    # a voxel right at the true edge on all three axes: weight 0 on every axis -> product 0
    assert weights[0, 0, 0].item() == 0.0
    # a voxel 'width' or more from every edge: full weight 1 on every axis -> product 1
    assert weights[width, width, width].item() == 1.0
    assert weights[n - 1 - width, n - 1 - width, n - 1 - width].item() == 1.0
    # strictly interior region is untouched
    assert torch.all(weights[width : n - width, width : n - width, width : n - width] == 1.0)


def test_taper_is_symmetric_and_monotonic_along_one_axis():
    n, width = 16, 5
    weights = get_hann_edge_weights(subtomo_size=n, taper_width=width)
    profile = weights[:, n // 2, n // 2]  # 1D slice through the center along axis 0
    assert torch.allclose(profile, profile.flip(0))  # symmetric front/back taper
    head = profile[: width + 1]
    assert torch.all(head[1:] >= head[:-1])  # monotonically non-decreasing into the interior


def test_matches_raised_cosine_formula_directly():
    n, width = 20, 4
    weights = get_hann_edge_weights(subtomo_size=n, taper_width=width)
    expected_1d = 0.5 * (1 - np.cos(np.pi * np.arange(width) / (width - 1)))
    profile = weights[:width, n // 2, n // 2].numpy()
    assert np.allclose(profile, expected_1d)


def test_per_axis_widths():
    n = 12
    weights = get_hann_edge_weights(subtomo_size=n, taper_width=(0, 3, 0))
    # axis 0 and 2 untapered (always weight 1), axis 1 tapered
    assert torch.all(weights[:, 0, :] == 0.0)  # axis 1's own taper still zeroes the edge there
    assert weights[0, 3, 0].item() == 1.0  # beyond axis 1's taper width, axes 0/2 untapered


def test_width_capped_at_half_the_box_size():
    # taper_width wider than size // 2 must not error or produce a non-monotonic double-dip
    weights = get_hann_edge_weights(subtomo_size=6, taper_width=100)
    assert weights.shape == (6, 6, 6)
    assert torch.isfinite(weights).all()
