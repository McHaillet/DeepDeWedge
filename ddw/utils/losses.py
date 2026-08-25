import torch

from .fourier import apply_fourier_mask_to_tomo


def data_consistency_loss(x_hat0, x_hat1, y0, y1, ctf):
    """
    Noise2Noise-style data-consistency loss. 'x_hat0'/'x_hat1' are the model's estimates
    from the two independent-noise raw observations 'y0'/'y1' (which already carry the same
    physical 'ctf' baked in from acquisition/reconstruction - it is never re-applied to them).
    Each estimate is re-masked with the canonical (native-orientation) 'ctf' and compared
    against the *other*, cross-wise, raw observation - not the one it came from, which would
    let the model trivially learn identity without ever averaging out noise.

    Because 'y0' and 'y1' share the exact same physical 'ctf', frequencies where 'ctf' is
    near zero contribute ~0 to both sides automatically (both the re-masked estimate and the
    raw target are ~0 there) - unlike the old two-region masked_loss, no extra region
    weighting is needed to handle this, which matters for a continuous (non-binary) CTF.
    """
    term0 = apply_fourier_mask_to_tomo(x_hat0, ctf) - y1
    term1 = apply_fourier_mask_to_tomo(x_hat1, ctf) - y0
    return term0.pow(2).mean() + term1.pow(2).mean()


def equivariance_loss(x_double_hat, target, mask, norm="ortho"):
    """
    Equivariant-imaging-style self-consistency loss, cross-paired the same way as
    data_consistency_loss: the caller rotates and re-masks (with the canonical, native-
    orientation 'ctf') one of the model's own estimates ("x_hat2") and passes it through the
    model a second time to produce 'x_double_hat', then rotates it back (R^-1) into the
    canonical frame before calling this function. 'target' is a rotated-then-unrotated - i.e.
    plain, canonical-frame - copy of the *other* estimate ("x_hat1"), not the one used to
    build the model's input. Comparing 'x_double_hat - target' after un-rotating (rather than
    un-rotating R^-1(x_double_hat - R(target))) gives the identical result since the grid
    rotations used here are linear and exactly invertible (see rotate_vol), so 'target' never
    actually needs to be rotated by the caller. Cross-pairing this way (instead of comparing
    'x_double_hat' back to a rotated copy of the same "x_hat2" it was built from) teaches the
    model to fill in 'ctf''s (anisotropic) null space by agreeing with its *other*,
    independent-noise estimate, rather than merely learning to undo its own rotation+ctf
    round-trip.

    'mask' (the same 'ctf' used everywhere else) is used purely as a Fourier-domain
    reliability weight for the comparison here - unlike the ctf application that builds the
    model's input upstream of this loss, it is never applied as a forward measurement
    operator. It's fine for 'mask''s own (rotation-invariant) zero-crossings to stay unfilled:
    there's no real data there in either representation, and weighting the comparison by
    'mask' means this loss doesn't force them to be recovered.

    Both 'x_double_hat' and 'target' must be constructed from detached estimates by the
    caller - a standard equivariant-imaging stop-gradient, not a limitation of rotate_vol
    (which is itself differentiable). This loss's gradient therefore only flows through the
    *second* application of the model (the one producing 'x_double_hat').

    The FFT uses the same rfftn/'norm="ortho"' convention as apply_fourier_mask_to_tomo. With
    an orthonormal transform, Parseval's theorem makes sum(|rfftn(diff)|^2) approximately half
    of sum(|diff|^2) (rfftn keeps only about half the full spectrum, dropping the redundant
    conjugate-symmetric half), and the rfftn grid has approximately half as many entries as
    the real-space volume too - so '.mean()' over the (masked) frequency-domain elements lands
    on the same scale as the real-space MSE used by data_consistency_loss, with no extra
    scaling factor needed.
    """
    diff_ft = torch.fft.rfftn(x_double_hat - target, dim=(-3, -2, -1), norm=norm)
    return (mask * diff_ft).abs().pow(2).mean()
