import math

import torch
from scipy import ndimage, spatial


def rotate_vol_around_axis(vol, rot_angle, rot_axis, output_shape=None, order=3, center=None):
    """
    Rotates the 3D tensor 'vol' by 'rot_angle' degrees around 'rot_axis'. The rotated tensor, which is typically larger than the original one, is center-cropped such that it has dimensions 'output_shape'. If 'output_shape' is None, the rotated tensor is cropped to the dimensions of 'vol'.

    'center' is the pivot point of the rotation, in (index-space) voxel coordinates for each of the 3 axes. If None (default), it is the geometric center of 'vol', (shape - 1) / 2 - the right pivot for real-space volumes. Fourier-domain arrays (masks/CTFs) should instead pivot on their DC voxel; use 'rotate_fourier_mask_around_axis' for those.
    """
    vol_shape = torch.tensor(vol.shape[-3:])
    if output_shape is None:
        output_shape = vol_shape
    # need later for cropping
    crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(vol_shape, output_shape)]
    if rot_angle != 0:
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle)
        rot_angle = torch.deg2rad(rot_angle)
        # convert rotation axis and angle to a 3x3 rotation matrix
        rot_axis = rot_axis.float()
        rot = spatial.transform.Rotation.from_rotvec(
            rot_angle * (rot_axis / rot_axis.norm())
        )
        rot_mat = rot.as_matrix()
        # determine offset to rotate around center of volume
        # see https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
        # -1 because indexing starts at 0
        if center is None:
            c_in = 0.5 * (vol_shape - torch.ones(3)).float().numpy()
        else:
            c_in = center
        offset = c_in - rot_mat @ c_in
        # apply the rotation using affine_transform
        vol = torch.tensor(
            ndimage.affine_transform(vol, matrix=rot_mat, offset=offset, order=order),
            device=vol.device,
            dtype=vol.dtype,
        )
    vol = vol[
        crop_offset[0] : crop_offset[0] + output_shape[0],
        crop_offset[1] : crop_offset[1] + output_shape[1],
        crop_offset[2] : crop_offset[2] + output_shape[2],
    ]
    return vol


def rotate_fourier_mask_around_axis(mask, rot_angle, rot_axis, output_shape=None):
    """
    Rotates the Fourier-domain mask/CTF 'mask' by 'rot_angle' degrees around 'rot_axis', pivoting on its DC (zero-frequency) voxel rather than the geometric array center that 'rotate_vol_around_axis' uses for real-space volumes.

    'mask' is assumed to follow this codebase's fftshift convention (see fft_3d/get_3d_fft_freqs_on_grid), which places DC at index N // 2 along each axis. That coincides with the geometric center (N - 1) / 2 only for odd N; for even N (the common case) the two are half a voxel apart, so rotating a mask with 'rotate_vol_around_axis' would pivot off of its true DC and introduce a systematic sub-voxel misalignment.

    Interpolation order is hard-coded to 1 (linear), unlike 'rotate_vol_around_axis's default of 3 (cubic spline): cubic interpolation can overshoot and produce negative values, which don't make sense for a mask/CTF whose values represent a confidence/attenuation in [0, 1].
    """
    mask_shape = torch.tensor(mask.shape[-3:])
    dc_center = (mask_shape // 2).float().numpy()
    return rotate_vol_around_axis(
        mask,
        rot_angle=rot_angle,
        rot_axis=rot_axis,
        output_shape=output_shape,
        order=1,
        center=dc_center,
    )
