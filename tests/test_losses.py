"""
Tests for ddw.utils.losses: data_consistency_loss and equivariance_loss. Pure CPU/torch -
no GPU needed.
"""
import torch

from ddw.utils.fourier import apply_fourier_mask_to_tomo
from ddw.utils.losses import (
    cross_consistency_loss,
    data_consistency_loss,
    equivariance_loss,
)


def test_data_consistency_loss_zero_for_perfect_cross_reconstruction():
    """
    If reconvolving x_hat with ctf exactly reproduces the (cross-wise) raw observation y,
    the loss must be exactly zero.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat = torch.randn(N, N, N)
    y = apply_fourier_mask_to_tomo(x_hat, ctf)
    loss = data_consistency_loss(x_hat, y, ctf)
    assert loss.item() < 1e-10


def test_data_consistency_loss_is_positive_otherwise():
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat = torch.randn(N, N, N)
    y = torch.randn(N, N, N)
    loss = data_consistency_loss(x_hat, y, ctf)
    assert loss.item() > 0


def test_data_consistency_loss_ignores_zero_ctf_frequencies():
    """
    Where ctf is exactly zero, the re-masked estimate is zero regardless of x_hat's value
    there, so the loss must not depend on x_hat at those frequencies as long as the raw
    observation agrees there too (also being subject to the same physical ctf). In rfftn
    convention every entry is independently maskable (no Hermitian-symmetry pairing to
    worry about, unlike the old full/fftshifted mask representation), so a perturbation with
    Fourier support exactly on ctf == 0 can be built directly.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.zeros(N, N, N // 2 + 1)
    ctf[:, :, : N // 4] = 1.0

    x_hat_a = torch.randn(N, N, N)
    y = torch.zeros(N, N, N)

    delta_freq = torch.fft.rfftn(torch.randn(N, N, N), norm="ortho") * (1 - ctf)
    delta = torch.fft.irfftn(delta_freq, s=(N, N, N), norm="ortho")
    # sanity check: delta must indeed be invisible to ctf
    assert apply_fourier_mask_to_tomo(delta, ctf).abs().max().item() < 1e-4

    x_hat_b = x_hat_a + delta
    loss_a = data_consistency_loss(x_hat_a, y, ctf)
    loss_b = data_consistency_loss(x_hat_b, y, ctf)
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_equivariance_loss_zero_when_identical():
    torch.manual_seed(0)
    mask = torch.rand(6, 6, 4).clamp(0, 1)
    x = torch.randn(4, 6, 6, 6)
    assert equivariance_loss(x, x, mask).item() == 0.0


def test_equivariance_loss_matches_manual_fourier_mse():
    torch.manual_seed(0)
    N = 8
    mask = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    a = torch.randn(4, N, N, N)
    b = torch.randn(4, N, N, N)
    diff_ft = torch.fft.rfftn(a - b, dim=(-3, -2, -1), norm="ortho")
    expected = (mask * diff_ft).abs().pow(2).mean()
    assert torch.allclose(equivariance_loss(a, b, mask), expected)


def test_equivariance_loss_ignores_zero_mask_frequencies():
    """
    Where 'mask' is exactly zero, that frequency must not contribute to the loss, regardless
    of how much 'x_double_hat' and 'target' disagree there - 'mask' is a comparison weight
    only, not applied as a forward operator to either side beforehand.
    """
    torch.manual_seed(0)
    N = 8
    mask = torch.zeros(N, N, N // 2 + 1)
    mask[:, :, : N // 4] = 1.0

    a = torch.randn(N, N, N)
    b = torch.randn(N, N, N)
    loss_ab = equivariance_loss(a, b, mask)

    delta_freq = torch.fft.rfftn(torch.randn(N, N, N), norm="ortho") * (1 - mask)
    delta = torch.fft.irfftn(delta_freq, s=(N, N, N), norm="ortho")
    loss_ab_perturbed = equivariance_loss(a + delta, b, mask)
    assert torch.allclose(loss_ab, loss_ab_perturbed, atol=1e-4)


def test_cross_consistency_loss_zero_when_identical():
    torch.manual_seed(0)
    ctf = torch.rand(6, 6, 4).clamp(0, 1)
    x = torch.randn(4, 6, 6, 6)
    assert cross_consistency_loss(x, x, ctf).item() == 0.0


def test_cross_consistency_loss_matches_manual_weighted_power_spectrum():
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x0_hat = torch.randn(4, N, N, N)
    x1_hat = torch.randn(4, N, N, N)
    diff_ft = torch.fft.rfftn(x0_hat - x1_hat, dim=(-3, -2, -1), norm="ortho")
    expected = ((1 - ctf**2) * diff_ft.abs().pow(2)).mean()
    assert torch.allclose(cross_consistency_loss(x0_hat, x1_hat, ctf), expected)


def test_cross_consistency_loss_ignores_frequencies_where_ctf_is_one():
    """
    Where 'ctf' is exactly 1, the (1-ctf^2) weight is exactly 0, so disagreement between
    x0_hat and x1_hat there must not affect the loss - dc_loss already fully constrains that
    frequency, so this term shouldn't also compete there.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.zeros(N, N, N // 2 + 1)
    ctf[:, :, : N // 4] = 1.0  # fully trusted band

    x0_hat = torch.randn(N, N, N)
    x1_hat = torch.randn(N, N, N)
    loss_a = cross_consistency_loss(x0_hat, x1_hat, ctf)

    delta_freq = torch.fft.rfftn(torch.randn(N, N, N), norm="ortho") * ctf
    delta = torch.fft.irfftn(delta_freq, s=(N, N, N), norm="ortho")
    loss_b = cross_consistency_loss(x0_hat + delta, x1_hat, ctf)
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_cross_consistency_loss_fully_active_where_ctf_is_zero():
    """
    Where 'ctf' is exactly 0, the (1-ctf^2) weight is exactly 1 - dc_loss gives that
    frequency ~no constraint, so this term should apply full weight there.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.zeros(N, N, N // 2 + 1)
    x0_hat = torch.randn(N, N, N)
    x1_hat = torch.randn(N, N, N)
    diff_ft = torch.fft.rfftn(x0_hat - x1_hat, dim=(-3, -2, -1), norm="ortho")
    expected = diff_ft.abs().pow(2).mean()
    assert torch.allclose(cross_consistency_loss(x0_hat, x1_hat, ctf), expected)


def test_equivariance_loss_scale_matches_real_space_mse():
    """
    With an all-ones 'mask' and 'norm="ortho"', the Fourier-domain loss lands on approximately
    the same scale as a plain real-space MSE (matching data_consistency_loss's scale) - see
    equivariance_loss's docstring on why rfftn + '.mean()' achieves this.
    """
    torch.manual_seed(0)
    N = 32
    mask = torch.ones(N, N, N // 2 + 1)
    a = torch.randn(N, N, N)
    b = torch.randn(N, N, N)
    fourier_loss = equivariance_loss(a, b, mask)
    real_space_mse = (a - b).pow(2).mean()
    assert torch.allclose(fourier_loss, real_space_mse, rtol=0.05)
