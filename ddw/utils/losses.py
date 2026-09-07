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
    residual_sq = (apply_fourier_mask_to_tomo(x_hat, ctf) - y).pow(2)
    return residual_sq.mean()


def equivariance_loss(x_double_hat, target):
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

    Plain MSE between 'x_double_hat' and 'target', matching the original equivariant-imaging
    formulation - no Fourier-domain CTF weighting.

    Both 'x_double_hat' and 'target' must be constructed from detached estimates by the
    caller - a standard equivariant-imaging stop-gradient, not a limitation of rotate_vol
    (which is itself differentiable). This loss's gradient therefore only flows through the
    *second* application of the model (the one producing 'x_double_hat').
    """
    return (x_double_hat - target).pow(2).mean()
