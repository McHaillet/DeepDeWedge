"""
Tests for ddw.utils.rotation, in particular rotate_fourier_mask_around_axis, which
pivots on the DC (zero-frequency) voxel - index N // 2 per axis, matching fft_3d's
fftshift convention - rather than the geometric array center (N - 1) / 2 that
rotate_vol_around_axis uses for real-space volumes. The two centers coincide only
for odd-sized grids; for even N (the common case) they differ by half a voxel.
"""
import torch

from ddw.utils.missing_wedge import get_missing_wedge_mask
from ddw.utils.rotation import rotate_fourier_mask_around_axis, rotate_vol_around_axis


def _impulse_at_dc(N):
    mask = torch.zeros(N, N, N)
    mask[N // 2, N // 2, N // 2] = 1.0
    return mask


def test_rotate_fourier_mask_around_axis_keeps_dc_fixed_for_even_size():
    """
    DC is the exact fixed point of the rotation once correctly pivoted, so an
    impulse placed there must come back out exactly 1.0 at that same index for any
    rotation angle/axis - not just approximately, and not smeared onto neighbors.
    """
    N = 8
    dc = (N // 2, N // 2, N // 2)
    mask = _impulse_at_dc(N)
    rot_axis = torch.tensor([1.0, 2.0, 3.0])
    for angle in [15, 45, 90, 137]:
        rotated = rotate_fourier_mask_around_axis(mask, rot_angle=angle, rot_axis=rot_axis)
        assert rotated[dc].item() == 1.0, (angle, rotated[dc].item())


def test_rotate_vol_around_axis_does_not_keep_dc_fixed_for_even_size():
    """
    Documents the bug rotate_fourier_mask_around_axis fixes: naively rotating a
    Fourier-domain mask with rotate_vol_around_axis (geometric-center pivot, half a
    voxel off DC for even N) smears the impulse at DC onto its neighbors instead of
    reproducing it exactly.
    """
    N = 8
    dc = (N // 2, N // 2, N // 2)
    mask = _impulse_at_dc(N)
    rot_axis = torch.tensor([1.0, 2.0, 3.0])
    rotated = rotate_vol_around_axis(mask, rot_angle=45, rot_axis=rot_axis, order=1)
    assert rotated[dc].item() < 1.0


def test_rotate_fourier_mask_around_axis_hardcodes_linear_and_avoids_negative_values():
    """
    Cubic spline interpolation (order=3, rotate_vol_around_axis's default) can
    overshoot past a sharp binary edge and produce negative/>1 values, which don't
    make sense for a mask/CTF representing a [0, 1] confidence. Linear interpolation
    can't overshoot a monotonic edge like this, so it must stay in [0, 1];
    rotate_fourier_mask_around_axis hard-codes it (no order option) rather than
    inheriting rotate_vol_around_axis's order=3 default.
    """
    N = 16
    mask = get_missing_wedge_mask(grid_size=[N, N, N], mw_angle=60).float()
    rot_axis = torch.tensor([1.0, 2.0, 3.0])

    # same DC-voxel pivot rotate_fourier_mask_around_axis uses internally, but with
    # cubic interpolation, to confirm this input actually triggers overshoot
    dc_center = (torch.tensor(mask.shape) // 2).float().numpy()
    rotated_cubic = rotate_vol_around_axis(
        mask, rot_angle=37, rot_axis=rot_axis, order=3, center=dc_center
    )
    assert rotated_cubic.min().item() < 0, "expected cubic interpolation to overshoot for this input"

    rotated = rotate_fourier_mask_around_axis(mask, rot_angle=37, rot_axis=rot_axis)
    assert rotated.min().item() >= 0
    assert rotated.max().item() <= 1


def test_rotate_fourier_mask_around_axis_matches_rotate_vol_around_axis_for_odd_size():
    """
    For odd-sized grids, DC (N // 2) and the geometric center ((N - 1) / 2) coincide,
    so both rotation functions must agree exactly.
    """
    N = 9
    torch.manual_seed(0)
    mask = torch.rand(N, N, N)
    rot_axis = torch.tensor([1.0, 2.0, 3.0])
    a = rotate_fourier_mask_around_axis(mask, rot_angle=37, rot_axis=rot_axis)
    b = rotate_vol_around_axis(mask, rot_angle=37, rot_axis=rot_axis, order=1)
    assert torch.allclose(a, b)
