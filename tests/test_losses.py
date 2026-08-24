"""
Tests for ddw.utils.losses: data_consistency_loss and equivariance_loss. Pure CPU/torch -
no GPU needed.
"""
import torch

from ddw.utils.fourier import apply_fourier_mask_to_tomo
from ddw.utils.losses import data_consistency_loss, equivariance_loss


def test_data_consistency_loss_zero_for_perfect_cross_reconstruction():
    """
    If reconvolving x_hat0/x_hat1 with ctf exactly reproduces the *other* raw observation,
    the loss must be exactly zero.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat0 = torch.randn(N, N, N)
    x_hat1 = torch.randn(N, N, N)
    y1 = apply_fourier_mask_to_tomo(x_hat0, ctf)
    y0 = apply_fourier_mask_to_tomo(x_hat1, ctf)
    loss = data_consistency_loss(x_hat0, x_hat1, y0, y1, ctf)
    assert loss.item() < 1e-10


def test_data_consistency_loss_is_positive_otherwise():
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat0 = torch.randn(N, N, N)
    x_hat1 = torch.randn(N, N, N)
    y0 = torch.randn(N, N, N)
    y1 = torch.randn(N, N, N)
    loss = data_consistency_loss(x_hat0, x_hat1, y0, y1, ctf)
    assert loss.item() > 0


def test_data_consistency_loss_ignores_zero_ctf_frequencies():
    """
    Where ctf is exactly zero, the re-masked estimate is zero regardless of x_hat's value
    there, so the loss must not depend on x_hat at those frequencies as long as the raw
    observations agree there too (both being subject to the same physical ctf). In rfftn
    convention every entry is independently maskable (no Hermitian-symmetry pairing to
    worry about, unlike the old full/fftshifted mask representation), so a perturbation with
    Fourier support exactly on ctf == 0 can be built directly.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.zeros(N, N, N // 2 + 1)
    ctf[:, :, : N // 4] = 1.0

    x_hat0_a = torch.randn(N, N, N)
    x_hat1 = torch.randn(N, N, N)
    y0 = torch.zeros(N, N, N)
    y1 = apply_fourier_mask_to_tomo(x_hat1, ctf)

    delta_freq = torch.fft.rfftn(torch.randn(N, N, N), norm="ortho") * (1 - ctf)
    delta = torch.fft.irfftn(delta_freq, s=(N, N, N), norm="ortho")
    # sanity check: delta must indeed be invisible to ctf
    assert apply_fourier_mask_to_tomo(delta, ctf).abs().max().item() < 1e-4

    x_hat0_b = x_hat0_a + delta
    loss_a = data_consistency_loss(x_hat0_a, x_hat1, y0, y1, ctf)
    loss_b = data_consistency_loss(x_hat0_b, x_hat1, y0, y1, ctf)
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_equivariance_loss_zero_when_identical():
    x = torch.randn(4, 6, 6, 6)
    assert equivariance_loss(x, x).item() == 0.0


def test_equivariance_loss_matches_manual_mse():
    torch.manual_seed(0)
    a = torch.randn(4, 6, 6, 6)
    b = torch.randn(4, 6, 6, 6)
    expected = (a - b).pow(2).mean()
    assert torch.allclose(equivariance_loss(a, b), expected)
