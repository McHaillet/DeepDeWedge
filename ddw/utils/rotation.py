import itertools
import random

import torch

# Sub-tomograms are stored axis-ordered (Z, Y, X) throughout this codebase (see e.g.
# scripts/reassemble_tomogram.py), and the tilt series is acquired by tilting around the Y
# axis - dim 1 of the last 3 dims.
TILT_AXIS = 1

BASE_SEED = 888


def get_grid_rotations(tilt_axis=TILT_AXIS):
    """
    Returns the 20 rotations that can be applied to a cubic sub-tomogram exactly, without any
    interpolation, and that actually change its missing-wedge/CTF direction relative to
    'tilt_axis'. Each rotation is a signed permutation matrix (a permutation of the 3 axes
    with independent +-1 sign flips) - the full set of 24 such matrices are exactly the
    proper (det=+1) rotations that map a cubic voxel grid exactly onto itself. Of those 24, 4
    only rotate the volume around 'tilt_axis' (by 0 or 180 degrees) or flip it along
    'tilt_axis' without otherwise touching the missing-wedge/CTF's orientation, so they leave
    it unchanged; the remaining 20 are returned here. This follows IsoNet (Supplementary Fig.
    2), which uses the same 20 grid-aligned rotations for interpolation-free data
    augmentation.
    """
    perp = [d for d in range(3) if d != tilt_axis]
    rotations = []
    for perm in itertools.permutations(range(3)):
        base = torch.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            mat = (base * torch.tensor(signs).reshape(3, 1)).to(torch.int64)
            if round(torch.linalg.det(mat.float()).item()) != 1:
                continue  # keep only proper rotations
            tilt_axis_fixed = mat[tilt_axis, tilt_axis] != 0
            perp_axes_swapped = mat[perp[0], perp[1]] != 0
            if tilt_axis_fixed and not perp_axes_swapped:
                continue  # leaves the missing-wedge/CTF direction unchanged
            rotations.append(mat)
    return rotations


GRID_ROTATIONS = get_grid_rotations()


def sample_grid_rotation(index, deterministic):
    """
    Samples one of the 20 grid rotations (see get_grid_rotations). If 'deterministic' is
    True, the choice only depends on 'index' (via BASE_SEED + index), giving reproducible
    rotations, e.g. for validation.
    """
    rng = random.Random(BASE_SEED + index) if deterministic else random
    return rng.choice(GRID_ROTATIONS)


def rotate_vol(vol, rot_mat):
    """
    Applies the grid rotation 'rot_mat' (one of get_grid_rotations()'s 20 signed permutation
    matrices) to the last 3 dims of 'vol' by permuting/flipping those axes. Since 'rot_mat'
    maps the voxel grid exactly onto itself, this is exact (no interpolation, unlike the old
    scipy.ndimage.affine_transform-based approach) and the output has exactly the same shape
    as the input - no cropping needed. Unlike that old approach, this is fully differentiable
    and needs no CPU round-trip.
    """
    lead = vol.dim() - 3
    # perm[i]: which of the 3 trailing input axes feeds output axis i
    perm = rot_mat.abs().argmax(dim=1).tolist()
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    flip_dims = [lead + i for i in range(3) if rot_mat[i, perm[i]] < 0]
    if flip_dims:
        vol = vol.flip(dims=flip_dims)
    return vol.permute(*range(lead), *(lead + p for p in inv_perm))
