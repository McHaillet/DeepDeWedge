"""
Tests for ddw.utils.fourier.rfft_mask_to_full_mask, the automatic conversion from
rfftn-convention (half-size last axis, DC at [0,0,0]) masks/CTFs to the full,
fftshifted (N, N, N) layout the rest of this codebase expects.
"""
import torch

from ddw.utils.fourier import fft_3d, ifft_3d, rfft_mask_to_full_mask


def test_already_full_shaped_mask_passes_through_unchanged():
    mask = torch.rand(8, 8, 8)
    assert torch.equal(rfft_mask_to_full_mask(mask), mask)


def test_matches_full_fftn_for_a_complex_spectrum():
    """
    rfftn(x) is exactly the first N//2+1 entries of the last axis of fftn(x). Expanding
    it back out (undoing the fftshift this function applies, to compare against the
    raw/unshifted fftn output) must reproduce fftn(x) exactly.
    """
    torch.manual_seed(0)
    for N in [8, 12, 32]:
        x = torch.randn(N, N, N)
        full_fftn = torch.fft.fftn(x)
        half = torch.fft.rfftn(x)
        reconstructed = torch.fft.ifftshift(rfft_mask_to_full_mask(half))
        assert torch.allclose(full_fftn, reconstructed, atol=1e-4), N


def test_masking_matches_direct_rfft_domain_masking():
    """
    The actual use case: a real-valued mask defined on the rfftn grid. Masking a
    volume with it (via irfftn(rfftn(vol) * mask_half, ...)) must give the same
    result as masking with the expanded+shifted mask via ddw's own fft_3d/ifft_3d.
    """
    torch.manual_seed(0)
    for N in [8, 12, 32]:
        mask_half = torch.rand(N, N, N // 2 + 1)
        mask_full = rfft_mask_to_full_mask(mask_half)
        vol = torch.randn(N, N, N)

        direct = torch.fft.irfftn(torch.fft.rfftn(vol) * mask_half, s=(N, N, N))
        via_ddw = ifft_3d(fft_3d(vol) * mask_full).real

        assert torch.allclose(direct, via_ddw, atol=1e-4), N


def test_output_shape_and_dc_at_center():
    N = 16
    mask_half = torch.ones(N, N, N // 2 + 1)  # trivial: everywhere trusted
    mask_full = rfft_mask_to_full_mask(mask_half)
    assert mask_full.shape == (N, N, N)
    # an all-ones rfftn-shaped mask expands to an all-ones full mask regardless of
    # where DC ends up, so check DC placement with a non-trivial mask instead
    mask_half = torch.zeros(N, N, N // 2 + 1)
    mask_half[0, 0, 0] = 1.0  # DC, unshifted convention
    mask_full = rfft_mask_to_full_mask(mask_half)
    assert mask_full[N // 2, N // 2, N // 2] == 1.0
    assert mask_full.sum() == 1.0
