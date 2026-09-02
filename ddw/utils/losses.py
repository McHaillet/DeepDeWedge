import torch

from .fourier import apply_fourier_mask_to_tomo


def data_consistency_loss(x_hat, y, ctf):
    """
    Noise2Noise-style data-consistency loss, for one estimate/observation pair. 'x_hat' is
    the model's estimate from one of the two independent-noise raw observations; 'y' is the
    *other* one (which already carries the same physical 'ctf' baked in from acquisition/
    reconstruction - it is never re-applied to it). 'x_hat' is re-masked with the canonical
    (native-orientation) 'ctf' and compared against 'y' - the caller is responsible for this
    cross-wise pairing (passing the observation 'x_hat' did *not* come from), since comparing
    an estimate to the observation it came from would let the model trivially learn identity
    without ever averaging out noise.

    Because both raw observations share the exact same physical 'ctf', frequencies where
    'ctf' is near zero contribute ~0 automatically (both the re-masked estimate and the raw
    target are ~0 there) - unlike the old two-region masked_loss, no extra region weighting
    is needed to handle this, which matters for a continuous (non-binary) CTF.

    LitUnet3D._step calls this once per step, with the (x_hat, y) pair picked at random
    between the two possible cross-wise pairings - see its docstring/comments for why.
    """
    return (apply_fourier_mask_to_tomo(x_hat, ctf) - y).pow(2).mean()


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


def cross_consistency_loss(x0_hat, x1_hat, ctf, norm="ortho"):
    """
    Direct full-band consistency between the model's two independent-noise estimates of the
    *same* physical region (one from subtomo0, one from subtomo1) - unlike
    data_consistency_loss/equivariance_loss, this compares the two estimates to each other
    directly, rather than each to a raw observation.

    Weighted by (1 - ctf^2), the complement of data_consistency_loss's own effective
    per-frequency weight: expanding that loss's squared residual '(ctf*x_hat - y)^2' shows its
    curvature w.r.t. x_hat scales as ctf^2, not ctf, so (1-ctf^2) is what sums to exactly 1
    with it at every frequency - strongest exactly where dc_loss's constraint is weakest
    (e.g. near the CTF's low-order zero-crossings), and near-zero where dc_loss already
    dominates, so this term can't fight or over-smooth the well-constrained high-frequency
    band (where forcing x0_hat/x1_hat to agree would just be classic Noise2Noise
    over-smoothing).

    The caller is responsible for breaking the shortcut where the model could satisfy this
    loss by reproducing the same fixed, position-anchored network artifact (e.g. conv
    boundary effects) on both sides rather than learning genuine shared content - typically by
    mirroring one side (a wedge-preserving grid flip - see
    ddw.utils.rotation.get_wedge_preserving_flips) before/after the forward pass that produced
    it - and for stop-gradient on one side (standard for a "make two views agree" loss, as
    with equivariance_loss), to prevent the pair from collapsing to an easy shared constant.
    """
    diff_ft = torch.fft.rfftn(x0_hat - x1_hat, dim=(-3, -2, -1), norm=norm)
    return ((1 - ctf**2) * diff_ft.abs().pow(2)).mean()
