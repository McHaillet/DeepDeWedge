import torch
from torch import fft


def fft_3d(tomo, norm="ortho"):
    """
    3D Fourier transform with fftshift.
    """
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


def ifft_3d(tomo, norm="ortho"):
    """
    Inverse 3D Fourier transform with fftshift.
    """
    fft_dim = (-1, -2, -3)
    return fft.ifftn(fft.ifftshift(tomo, dim=fft_dim), dim=fft_dim, norm=norm)


def apply_fourier_mask_to_tomo(tomo, mask, output="real"):
    """
    Multiplies the Fourier transform of 'tomo' with 'mask. This function is used to add the artificial missing wedges to the model inputs.
    """
    tomo_ft = fft_3d(tomo)
    tomo_ft_masked = tomo_ft * mask
    vol_filt = ifft_3d(tomo_ft_masked)
    if output == "real":
        return vol_filt.real
    elif output == "complex":
        return vol_filt


def rfft_mask_to_full_mask(mask):
    """
    Converts a real-valued Fourier mask/CTF in rfftn convention - shape
    (N, N, N//2+1), DC at index [0,0,0] - into the full (N, N, N) shape expected
    everywhere else in this codebase, with DC shifted to the center voxel (matching
    fft_3d's convention). If 'mask' is already shaped (N, N, N), it is returned
    unchanged.

    The expansion uses the symmetry X[i,j,k] = conj(X[(-i)%N, (-j)%N, N-k]) (a no-op
    for real-valued masks/dtypes, but kept so this is also correct if 'mask' happens
    to be stored with a complex dtype).
    """
    n0, n1, n2 = mask.shape[-3:]
    if n0 != n1 or n2 != n0 // 2 + 1:
        # not rfftn-shaped (assumed already full (N, N, N))
        return mask
    N = n0
    full = torch.zeros(mask.shape[:-3] + (N, N, N), dtype=mask.dtype, device=mask.device)
    full[..., :n2] = mask
    for k in range(n2, N):
        mirror_k = N - k
        mirrored = torch.flip(mask[..., mirror_k], dims=[-2, -1])
        mirrored = torch.roll(mirrored, shifts=(1, 1), dims=(-2, -1))
        full[..., k] = torch.conj(mirrored)
    return torch.fft.fftshift(full, dim=(-3, -2, -1))


def get_3d_fft_freqs_on_grid(grid_size, device="cpu"):
    """
    Produces a 3D tensor with shape 'grid_size' whose entries are the spatial frequencies that correspond to the entries of a fourier transform computed with 'fft_3d'.
    """
    z = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[0]), device=device))
    y = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[1]), device=device))
    x = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[2]), device=device))
    grid = torch.cartesian_prod(z, y, x)
    return grid
