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


def equivariance_loss(x_double_hat, x_hat_rotated):
    """
    Equivariant-imaging-style self-consistency loss. 'x_hat_rotated' is a rotated copy of one
    of the model's own estimates, re-masked with the canonical 'ctf' and passed through the
    model a second time to produce 'x_double_hat'. Matching these teaches the model to fill
    in 'ctf''s (anisotropic) null space, without ever rotating 'ctf' itself - only real-space
    volumes are rotated, since rotating a continuous CTF/mask array by interpolation (as the
    old rot_mw_mask scheme did) is itself an approximation. It's fine for 'ctf''s own
    (rotation-invariant) zero-crossings to stay unfilled: there's no real data there in
    either representation, and this loss doesn't force them to be recovered.

    Rotation is not differentiable (see rotate_vol_around_axis, which uses
    scipy.ndimage.affine_transform), so 'x_hat_rotated' must be constructed from a detached
    estimate by the caller. This loss's gradient therefore only flows through the *second*
    application of the model (the one producing 'x_double_hat'), training it to be
    equivariant with respect to its own (stop-gradient) estimate.
    """
    return (x_double_hat - x_hat_rotated).pow(2).mean()
