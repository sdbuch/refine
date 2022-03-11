#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with optimization-based registration solvers (in pytorch) 
This module will be written with self-contained helper functions (in contrast to
other modules like image_deformations), due to the common pytorch dependency.

To be called externally.

"""

# imports
import numpy as np
import numpy.linalg as npla
from numpy.random import default_rng
import scipy as sp
import matplotlib.pyplot as plt
import torch
from torch import tensor

def device():
    """Helper function to get a relevant device string (cpu or gpu) w/ pytorch

    This snippet can be pasted in various locations in the code where we want
    things to work whether they are on CPU or GPU.

    """

    import torch

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    return device

def precision():
    """Helper function to get the desired floating point type for computations

    """

    # Spike registration with 1x1 spikes doesn't work with fp32.
    # Might work with larger spikes though (losses aren't so small)
    fp32 = torch.float32
    fp64 = torch.float64
    return fp64

def resample_chunked(Y, tau_pt, chunk_sz):
    """Call grid sample with input data, but supported chunked broadcasting

    This is a workaround for the fact that grid_sample in pytorch does not
    support broadcasting. See pointers in discussion below.

    The workaround we will use here is to process grid_sample in the specified
    chunk size.

    Call conventions:
    1. Y is a single scene (1 x C x M x N).
    2. tau_pt is a batch of grids to sample on. (B x m x n x 2)
    3. chunk_sz is a valid chunk size (less than B): the grid samples are
        processed in batches of size chunk_sz. (in case B is too large to fit in
        GPU memory -- see below notes).

    Observations:
    1. Seems to be marginally slower at large-ish chunk sizes (batch_sz 64,
    chunk_sz 32)?
    2. grid_sample itself seems to be extremely slow with large motifs (not sure
    if there is a special issue when motif size exceeds input size / boundary
    condition issues??)

    Notes on batched interpolations:
    current (1.10.1?) pytorch version of grid_sample doesn't support batch
    dimension broadcasting. our use case is to do a bunch of different
    interpolations of small size in one image; availability of broadcasting
    would mean the grid_sample call would support this. as it stands, we need to
    copy the scene (batch_sz) times to get a fast interpolation, which is not
    memory-efficient. (as long as everything fits in memory, not a problem,
    though?)
    Needs more testing to see how close we come to the limit.
    The issue tracker is here: https://github.com/pytorch/pytorch/issues/2732

    """

    from torch.nn.functional import grid_sample

    batch_sz = tau_pt.shape[0]
    C = Y.shape[1]
    dev = device()

    output = torch.zeros((batch_sz, C) + tau_pt.shape[1:3], device=dev,
            dtype=precision())

    # Chunked operation
    for batch_idx in range(0, batch_sz, chunk_sz):
        output[batch_idx:batch_idx+chunk_sz, ...] = grid_sample(
                Y.expand(min(chunk_sz, batch_sz-batch_idx),-1,-1,-1),
                tau_pt[batch_idx:batch_idx+chunk_sz, ...], mode='bicubic',
                align_corners=True)

    return output

def reg_l2_rigid(Y_scene, X, mask, locs, sigma=3, sigma_scene=1.5,
        step=tensor(1e0, device=device(), dtype=precision()), max_iter=100,
        init_data=None, image_type='textured', rejection_thresh=1e-2,
        erode=False):
    """Registration of deformed-known-motif with GD on l2 loss + rigid motion

    Currently supporting two modes:
    1. cost smoothing objective (for textured motifs)
    2. image smoothing objective (for spike registration or common background)

    inputs:

    Y_scene : C x M x N  big scene to detect in
    X : C x m x n  motif to detect
    mask : 1 x m x n  {0, 1} mask for the motif X (tight bounding box)
        Can pass None if image_type == 'spike'
    locs : batch_sz x 2 matrix of stride locations to register at. 
           (!!) coordinates are (u, v) (!!)
    init_data : 2 element tuple or None
        Default is None -- then the initialization is done at the identity. If
        this is a 2 element tuple, then the first element is used for phi init
        (should be (batch_sz,) shaped tensor) and the second element is used for
        b init (should be (batch_sz, 2) shaped tensor)
    image_type : string, either 'textured' or 'spike'
        This specifies whether the input is a spike image or a textured image.
        - If 'textured', the registration run is a cost-smoothed formulation
          with scene smoothing.
        - If 'spike', the registration run is an image-smoothed formulation with
          no scene smoothing.
        It makes sense to call option 'spike' with common-background
        registration setups as well (image smoothing seems to be much more
        trustworthy when its assumptions are in play).

    outputs:
    
    phi : (batch_sz,) size tensor
        learned rotation parameters
    b : (batch_sz,2) size tensor
        learned translation parameters. In (v, u) coordinates
    err : (batch_sz, max_iter) size tensor
        Optimization objective errors for each batch

    Notes on torch convs:
    right way to replicate a "symmetric padded convolutional filtering" with
    boundary blowup is to:
     1. use an odd-size filter kernel
     2. the convs are actually cross-correlation, so fftshift the filter if it
       was centered at (0, 0) to begin with (otherwise, center it at the
       middle pixel)
     3. conv2d pads the front/back of the input by the same amount, rather
       than all on the back. so with half-width W = (filter_size - 1)/2, pad
       conv input X by 2*half-width (a 1x half-width padding is like "same"
       output size constraint: boundary effects are thrown away) [i.e. pass
       this parameter to conv2d padding]
     4. for separable convolution (each of C input channels conv'd separately
       with the same filter), copy the filter into a C x 1 x filter_size x
       filter_size tensor, and pass groups=C to conv2d


    """

    from torch.nn.functional import conv2d, grid_sample
    from registration_np import (gaussian_filter_2d)
    from scipy.fft import fftshift
    from images import (imagesc, get_affine_grid_basis, show_false_color)

    # OVERHEAD
    dev = device()
    C, M, N = Y_scene.shape
    c, roi_u, roi_v = X.shape
    batch_sz = locs.shape[0]
    # calculate chunk size automatically, as large as possible
    # heuristic: we seem to be able to grid sample a scene that takes up 4GB
    #   after copying
    chunk_sz = 2**int(np.floor(29 - np.log2(C * M * N)))
    chunk_sz = min(chunk_sz, batch_sz)

    # Filters setup
    if image_type == 'textured':
        G_scene_tail = int(np.ceil(3*sigma_scene))
        G_scene = fftshift(gaussian_filter_2d(2*G_scene_tail +1,
          N=2*G_scene_tail+1, sigma_u=sigma_scene))
        G_scene_pt = torch.tile(tensor(G_scene[None, ...], device=dev), (C, 1,
            1))[:, None, ...]
        G_tail = int(np.ceil(3*sigma))
        G = fftshift(gaussian_filter_2d(2*G_tail +1, N=2*G_tail+1, sigma_u=sigma))
        G_pt = torch.tile(tensor(G[None, ...], device=dev), (C, 1, 1))[:, None, ...]
    else:
        G_scene_tail = int(np.ceil(3*sigma_scene))
        G_scene = fftshift(gaussian_filter_2d(2*G_scene_tail +1,
          N=2*G_scene_tail+1, sigma_u=sigma_scene))
        G_scene_pt = torch.tile(tensor(G_scene[None, ...], device=dev), (C, 1,
            1))[:, None, ...]
        # Complementary smoothing approach
        sigma_compl = np.sqrt(sigma**2 - sigma_scene**2)
        G_tail = int(np.ceil(3*sigma_compl))
        G = fftshift(gaussian_filter_2d(2*G_tail +1, N=2*G_tail+1,
            sigma_u=sigma_compl))
        G_pt = torch.tile(tensor(G[None, ...], device=dev), (C, 1, 1))[:, None, ...]

    # Initialization setup (batched)
    if image_type == 'textured':
        pad_sz = 2 * G_tail
    else:
        # For image smoothing, there is more smoothing in the gradient due to
        # the convolution adjoint
        pad_sz = 4 * G_tail + 2 * G_scene_tail

    (basis_u, basis_v, basis_shift, shift_scale) = get_affine_grid_basis(M, N,
            roi_u+pad_sz, roi_v+pad_sz, locs=locs - torch.tensor(((pad_sz//2,
                pad_sz//2),), device=dev, dtype=precision()))

    m_vec = torch.arange(roi_u + pad_sz, device=dev, dtype=precision())
    m_vec -= pad_sz//2 + (roi_u-1)/2
    n_vec = torch.arange(roi_v + pad_sz, device=dev, dtype=precision())
    n_vec -= pad_sz//2 + (roi_v-1)/2

    if init_data is None:
        # Default (identity) initialization
        # Matrix parameters
        b = torch.tile(torch.zeros((2,), device=dev, dtype=precision())[None,
            ...], (batch_sz, 1))
        phi = torch.tile(torch.zeros((1,), device=dev,
            dtype=precision())[None, ...], (batch_sz, 1))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)

        # Flip the u and v tensors for grid_sample compat
        tau_pt = (torch.cat((sinphi[...,None,None] * basis_u +
            cosphi[...,None,None] * basis_v, cosphi[...,None,None] *
            basis_u + -sinphi[...,None,None] * basis_v), axis=-1) +
            basis_shift + (b * shift_scale)[:, None, None, :])
    else:
        # passthrough initialization
        phi = init_data[0]
        b = init_data[1]
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        tau_pt = (torch.cat((sinphi[...,None,None] * basis_u +
            cosphi[...,None,None] * basis_v, cosphi[...,None,None] *
            basis_u + -sinphi[...,None,None] * basis_v), axis=-1) +
            basis_shift + (b * shift_scale)[:, None, None, :])

    # TODO: This is a bug, fix this... See the scheme in reg_l2_spike for the
    # fix
    if image_type == 'textured':
        # Perform scene smoothing
        Y = conv2d(Y_scene[None, ...], G_scene_pt, padding=G_scene_tail, groups=C)
    else:
        Y = conv2d(Y_scene[None, ...], G_scene_pt, padding=G_scene_tail, groups=C)

    # Prepare the cost function and smoothing-mode-specific internals
    if image_type == 'textured':
        mask_smeared = conv2d(mask[None, ...], G_pt[0,...][None,:],
                padding=2*G_tail, groups=1)
        term3 = torch.sum(G_pt[0,...]) * torch.sum((X * mask) ** 2)
        X_smeared = conv2d(X[None, ...], G_pt, padding=2*G_tail, groups=C)
        cost = lambda Y: 0.5 * (torch.sum(Y**2 * mask_smeared, axis=(-1,-2,-3)) 
                - 2 * torch.sum(Y * X_smeared,axis=(-1, -2, -3)) + term3)
        residual_func = lambda Y: mask_smeared * Y - X_smeared
    else:
        mask = torch.zeros((1, pad_sz + roi_u, pad_sz + roi_v), device=dev,
                dtype=precision())
        mask[0, G_tail:roi_u+3*G_tail+2*G_scene_tail:,
                G_tail:roi_v+3*G_tail+2*G_scene_tail:] = torch.ones(
                (roi_u + 2*G_tail+2*G_scene_tail, roi_v +
                    2*G_tail+2*G_scene_tail), device=dev, dtype=precision())
        X_1conv = conv2d(X[None, ...], G_scene_pt, padding=2*G_tail +
                2*G_scene_tail, groups=C)
        X_1conv = conv2d(X_1conv, G_pt, padding=G_tail, groups=C)
        # Check for rejections
        # A bit extra on the first iteration for savings...
        cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
        y_conv = conv2d(cur_Y, G_pt, padding=G_tail, groups=C)
        normalized_corrs = (torch.sum(X_1conv * y_conv, axis=(1,2,3)) /
                torch.sum(X_1conv**2, axis=(1,2,3))**0.5 /
                torch.sum(y_conv**2, axis=(1,2,3))**0.5)
        accept_idxs = torch.where(normalized_corrs >
                rejection_thresh)[0]
        # Only process the accepted strides.
        accept_sz = len(accept_idxs)
        print(f'{accept_sz} accepted strides (rejecting at level'
                f' {rejection_thresh})')
        # re-initialize
        if accept_sz < 2:
            print('no accepted indices')
        phi = phi[accept_idxs, ...]
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        b = b[accept_idxs, ...]
        tau_pt = tau_pt[accept_idxs, ...]
        basis_shift = basis_shift[accept_idxs, ...]

    # Prepare errors tensor and cost
    error = float('inf') * torch.ones( (batch_sz, max_iter), device=dev,
            dtype=precision())

    # Prepare preconditioner
    motif_rad = np.max((roi_u/2, roi_v/2))
    A_scale = torch.tensor(motif_rad**2 / sigma, device=dev,
            dtype=precision())
    b_scale = torch.tensor(motif_rad / sigma, device=dev, dtype=precision())

    # Main loop!
    # Get differentiable object ... maybe need to rewrite this downstream
    for idx in range(max_iter-1):
        # Interpolation and cost computation
        tau_pt.requires_grad = True
        cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
        if image_type == 'textured':
            residual = residual_func(cur_Y)
            error[:, idx] = cost(cur_Y.detach())
        else:
            y_conv = conv2d(cur_Y.detach(), G_pt, padding=G_tail, groups=C)
            f_res = (y_conv - X_1conv)
            fw_res = mask[None,...] * f_res
            error[accept_idxs, idx] = 0.5 * torch.sum(fw_res**2, axis=(-1,-2,-3))
            residual = conv2d(fw_res, G_pt, padding=G_tail, groups=C)
        # Get tau gradients
        tau_dots = torch.autograd.grad(cur_Y, tau_pt, grad_outputs=residual)[0]
        # Do summations
        tau_dots_vsums = torch.sum(tau_dots, axis=(2,))
        tau_dots_usums = torch.sum(tau_dots, axis=(1,))
        # Contract
        # TODO: If the apparently Y_scene issue above is fixed, these need to be
        # fixed too (they should just use shift_scale instead of hard-coded
        # scales?)
        m_contractions = torch.einsum('abc,b -> ac', tau_dots_vsums,
                2/(M-1)*m_vec)
        n_contractions = torch.einsum('abc,b -> ac', tau_dots_usums,
                2/(N-1)*n_vec)
        # Combine with phi weights for phi grad
        combo_weights = torch.cat((cosphi, -1 * sinphi, -1 * sinphi, -1 * cosphi), dim=-1)
        grad_phi = torch.sum( torch.cat((m_contractions, n_contractions),
            dim=-1) * combo_weights, axis=(-1,)) / A_scale
        # b grad
        grad_b = (torch.sum(tau_dots_usums, axis=(1,)) * shift_scale /
                b_scale)

        # Gradient steps
        phi -= step * grad_phi[:, None]
        b -= step * grad_b
        # tau recalculating
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        tau_pt = (torch.cat((sinphi[...,None,None] * basis_u +
            cosphi[...,None,None] * basis_v, cosphi[...,None,None] *
            basis_u + -sinphi[...,None,None] * basis_v), axis=-1) +
            basis_shift + (b * shift_scale)[:, None, None, :])

    # Compute cost for last loop iteration
    cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
    if image_type == 'textured':
        error[:, max_iter-1] = cost(cur_Y.detach())
        b_out = b
        phi_out = phi
    else:
        f_res = (conv2d(cur_Y.detach(), G_pt, padding=G_tail, groups=C) -
                X_1conv)
        fw_res = mask[None,...] * f_res
        error[accept_idxs, max_iter-1] = 0.5 * torch.sum(fw_res**2, axis=(-1,-2,-3))
        # fill in the cost on the reject indices
        reject_mask = torch.ones((batch_sz,), device=dev, dtype=bool)
        reject_mask[accept_idxs] = False
        reject_idxs = torch.arange(batch_sz)[reject_mask]
        error[reject_idxs, :] = error[accept_idxs,0].max()
        # default settings for the rejected final transformation params
        b_out = torch.tile(torch.zeros((2,), device=dev,
            dtype=precision())[None, ...], (batch_sz, 1))
        phi_out = torch.tile(torch.zeros((1,), device=dev,
            dtype=precision())[None, ...], (batch_sz, 1))
        b_out[accept_idxs, :] = b
        phi_out[accept_idxs, :] = phi

    # Done
    return phi_out, b_out, error

def reg_l2_spike(Y_scene, X, locs, ctrs=None, sigma=10, sigma0=2,
        step_A=tensor(1e0, device=device(), dtype=precision()),
        step_b=tensor(1e0, device=device(), dtype=precision()),
        max_iter=100,
        init_data=None, rejection_thresh=1e-3, external_smoothing=False,
        record_process=False, quiet=True):
    """Complementary smoothing registration formulation with inv-affine motion

    NOTE: The filtering scheme used here doesn't explicitly normalize the
    filters to have unit ell_1 norm -- just uses the definition of the
    continuous pdf + discretizes it. So depending on corners cut for speed (see
    internal parameter tail_factor), at small values of sigma/sigma0 there may be
    issues of scale caused by this

    """

    from torch.nn.functional import conv2d, pad
    from images import (imagesc, get_affine_grid_basis, show_false_color,
            gaussian_cov, get_affine_grid_from_basis, gaussian_cov_adaptive,
            gaussian_cov_adaptive_dot, gaussian_cov_adaptive_xy,
            get_affine_grid_from_basis_xy, mask_to_bbox_pt)
    sfc = lambda X: show_false_color(X,normalize=True)

    # OVERHEAD
    dev = device()
    # torch.backends.cudnn.benchmark = True
    C, M, N = Y_scene.shape
    c, roi_u, roi_v = X.shape
    # This is easier to code for odd-size motifs, so just require it (can
    # always just pad). but it's not necessary (adjoint operator just needs to
    # be implemented differently...)
    if roi_u % 2 == 0:
        raise ValueError('The width and height of the motif X need to be odd' 
                " (for compatibility with pytorch conv2d's symmetric padding)."
                " Can enforce this by padding externally.")
    if roi_v % 2 == 0:
        raise ValueError('The width and height of the motif X need to be odd' 
                " (for compatibility with pytorch conv2d's symmetric padding)."
                " Can enforce this by padding externally.")
    batch_sz = locs.shape[0]
    # calculate chunk size automatically, as large as possible
    # heuristic: we seem to be able to grid sample a scene that takes up 4GB
    #   after copying (12GB mem capacity card)
    chunk_sz = 2**int(np.floor(29 - np.log2(C * M * N)))
    chunk_sz = min(chunk_sz, batch_sz)
    if ctrs is not None:
        # Centers mode: ignore locs and use centers instead to do the
        # transforming...
        ctrs_mode = True
    else:
        ctrs_mode = False

    # Basic (non-adaptive) filters setup
    tail_factor = 2
    if external_smoothing == False:
        # local scene filter
        G_scene_tail = int(np.ceil(tail_factor*sigma0))
        G_scene_pt = gaussian_cov(2*G_scene_tail+1, 2*G_scene_tail+1,
                Sigma=sigma0**2 * torch.eye(2,device=dev,dtype=precision()))[None,
                        None, ...].expand(C, -1, -1, -1)
    else:
        # External smoothing mode: we already applied a sigma0 filtering
        G_scene_tail = 0
    # Full-size filter
    if external_smoothing == False:
        G_tail = int(np.ceil(tail_factor*sigma))
        G_pt = gaussian_cov(2*G_tail+1, 2*G_tail+1,
                Sigma=sigma**2 * torch.eye(2,device=dev,dtype=precision()))[None,
                        None, ...].expand(C, -1, -1, -1)
    else:
        # External smoothing mode, we need to compensate for external smoothing
        #sigma_reduced = torch.sqrt(torch.tensor(sigma**2 - sigma0**2, device=dev,
        #        dtype=precision()))
        sigma_reduced = np.sqrt(sigma**2 - sigma0**2)
        G_tail = int(np.ceil(tail_factor*sigma_reduced))
        G_pt = gaussian_cov(2*G_tail+1, 2*G_tail+1, Sigma=sigma_reduced**2 *
                torch.eye(2,device=dev,dtype=precision()))[None, None,
                        ...].expand(C, -1, -1, -1)

    # Scene smoothing
    if external_smoothing == False:
        Y = conv2d(Y_scene[None, ...], G_scene_pt, padding=2*G_scene_tail,
                groups=C)
    else:
        # External smoothing mode, we already smoothed the scene
        Y = Y_scene[None, ...]

    # Initialization setup (batched)
    pad_sz = 4 * G_tail + 2 * G_scene_tail
    if ctrs_mode == False:
        # normal 'orthobasis' mode
        (basis_u, basis_v, basis_shift, shift_scale) = get_affine_grid_basis(M +
                2*G_scene_tail, N + 2*G_scene_tail, roi_u+pad_sz, roi_v+pad_sz,
                locs=locs - torch.tensor(((2*G_tail, 2*G_tail),), device=dev,
                    dtype=precision()))
        m_vec = torch.arange(roi_u + pad_sz, device=dev, dtype=precision())
        m_vec -= pad_sz//2 + (roi_u-1)/2
        n_vec = torch.arange(roi_v + pad_sz, device=dev, dtype=precision())
        n_vec -= pad_sz//2 + (roi_v-1)/2
    else:
        # offset center mode: correlated basis
        (basis_u, basis_v, basis_shift, shift_scale) = get_affine_grid_basis(M +
                2*G_scene_tail, N + 2*G_scene_tail, roi_u+pad_sz, roi_v+pad_sz,
                locs=-pad_sz//2*torch.ones_like(ctrs, device=dev,
                    dtype=precision()), ctrs=ctrs)
        m_vec = torch.arange(roi_u + pad_sz, device=dev, dtype=precision())
        m_vec -= pad_sz//2 + ctrs[0,0]
        n_vec = torch.arange(roi_v + pad_sz, device=dev, dtype=precision())
        n_vec -= pad_sz//2 + ctrs[0,1]

    if init_data is None:
        # Default (identity) initialization
        # Matrix parameters
        A = torch.eye(2, device=dev, dtype=precision())[None,
                ...].expand(batch_sz, -1, -1)
        b = torch.zeros((batch_sz, 2), device=dev, dtype=precision())
        Ainv = torch.linalg.inv(A)
        Ainv_b = torch.bmm(Ainv, b[...,None])[..., 0]
    else:
        # passthrough initialization
        A = init_data[0]
        b = init_data[1]
        Ainv = torch.linalg.inv(A)
        Ainv_b = torch.bmm(Ainv, b[...,None])[..., 0]

    tau_pt = get_affine_grid_from_basis_xy(Ainv, -Ainv_b, basis_u, basis_v,
            basis_shift, shift_scale)

    # Create mask and filtered motif
    mask = torch.zeros((1, pad_sz + roi_u, pad_sz + roi_v), device=dev,
            dtype=precision())
    mask[0, G_tail:roi_u+3*G_tail+2*G_scene_tail:,
            G_tail:roi_v+3*G_tail+2*G_scene_tail:] = torch.ones(
            (roi_u + 2*G_tail+2*G_scene_tail, roi_v +
                2*G_tail+2*G_scene_tail), device=dev, dtype=precision())
    X_1conv = conv2d(X[None, ...], G_pt, padding=3*G_tail + G_scene_tail,
            groups=C)

    # Check for rejections
    # A bit extra on the first iteration for savings...
    G_adap = gaussian_cov_adaptive_xy(Ainv, M=2*G_tail+1, sigma=sigma,
            sigma0=sigma0)
    cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
    y_conv = conv2d(cur_Y, G_adap[None,...].expand(C,-1,-1,-1), padding=G_tail,
            groups=C)
    normalized_corrs = (torch.sum(X_1conv * y_conv, axis=(1,2,3)) /
            torch.sum(X_1conv**2, axis=(1,2,3))**0.5 /
            torch.sum(y_conv**2, axis=(1,2,3))**0.5)
    accept_idxs = torch.where(normalized_corrs >
            rejection_thresh)[0]
    # Only process the accepted strides.
    accept_sz = len(accept_idxs)
    print(f'{accept_sz} accepted strides (rejecting at level'
            f' {rejection_thresh})')
    # re-initialize
    A = A[accept_idxs, ...]
    b = b[accept_idxs, ...]
    Ainv = Ainv[accept_idxs, ...]
    Ainv_b = Ainv_b[accept_idxs, ...]
    tau_pt = tau_pt[accept_idxs, ...]
    basis_shift = basis_shift[accept_idxs, ...]

    # Prepare errors tensor and cost
    error = float('inf') * torch.ones( (batch_sz, max_iter), device=dev,
            dtype=precision())

    if record_process:
        A_list = torch.zeros(max_iter, A.shape[1], A.shape[2], device=dev,
                dtype=precision())
        b_list = torch.zeros(max_iter, b.shape[1], device=dev,
                dtype=precision())
        A_list[0,:,:] = A
        b_list[0,:] = b
        
        Rvals = torch.zeros(max_iter, device=dev, dtype=precision())
        
    # Prepare preconditioner
    # Using motif_rad as a heuristic for the 1->2 norm of matrix of spike locs
    # (plus a scaling by root-num-channels)
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    bb_l, bb_r, bb_t, bb_b = mask_to_bbox_pt(X, thresh=1e-6)
    sz_u = bb_b - bb_t
    sz_v = bb_r - bb_l
    motif_rad = torch.max(sz_u/2, sz_v/2)
    overall_scale = 8*pi*sigma**4
    A_scale = overall_scale / motif_rad**2
    b_scale = overall_scale

    # Main loop!
    # Overhead definitions (never change)
    Id = torch.eye(2,device=dev, dtype=precision())
    m_vec_filt = (torch.arange(2*G_tail+1, device=dev, dtype=precision()) -
            (2*G_tail+1)//2)
    n_vec_filt = (torch.arange(2*G_tail+1, device=dev, dtype=precision()) -
            (2*G_tail+1)//2)
    grid = torch.cat((torch.kron(n_vec_filt, torch.ones(len(m_vec_filt),
        device=dev, dtype=precision()))[:,None],
        torch.kron(torch.ones(len(n_vec_filt), device=dev, dtype=precision()),
            m_vec_filt)[:,None]), dim=-1)
    for idx in range(max_iter-1):
        if not quiet:
            print(idx)
        
        # Create the adaptive filter
        # TODO: This code is not working right now for selective striding...
        # need to fix some uses of batch_sz below
        # Overhead for filters
        AAt = torch.bmm(A, torch.transpose(A,-1,-2))
        Sigma = (sigma**2 * Id[None,...].expand(batch_sz,2,2) - sigma0**2 * AAt)
        Sigmainv = torch.linalg.inv(Sigma)
        # Creating filter 
        exponent = -0.5 * (torch.sum(torch.bmm(Sigmainv,
            grid.T[None,...].expand(batch_sz,-1,-1)) * grid.T[None,...], dim=-2) +
            torch.logdet(Sigma)[...,None])
        G_adap = torch.transpose(0.5/pi * torch.exp(torch.reshape(exponent,
            (batch_sz, 2*G_tail+1, 2*G_tail+1))) /
            torch.sqrt(torch.det(AAt))[:,None,None], -1, -2)

        # Interpolation and cost computation
        tau_pt.requires_grad = True
        cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
        y_conv = conv2d(cur_Y.detach(), G_adap[None,...].expand(C,-1,-1,-1),
                padding=G_tail, groups=C)
        f_res = (y_conv - X_1conv)
        fw_res = mask[None,...] * f_res
        error[accept_idxs, idx] = 0.5/C * torch.sum(fw_res**2, axis=(-1,-2,-3))

        # Two filtered residuals (for jvp's): filter with G_adap, and with cur_Y
        residual_0 = 1/C * conv2d(fw_res, G_adap[None,...].expand(C,-1,-1,-1),
                padding=G_tail, groups=C)
        ## Channelwise cross-cor by collapsing dims
        residual_1 = 1/C * conv2d(
                fw_res.reshape(1, batch_sz * C, roi_u + pad_sz, roi_v + pad_sz),
                cur_Y.detach().reshape(batch_sz*C, 1, roi_u + pad_sz, roi_v +
                    pad_sz), padding=G_tail,
                groups=batch_sz*C).reshape(batch_sz, C, 2*G_tail+1, 2*G_tail+1)
        # Get tau gradients
        tau_dots = torch.autograd.grad(cur_Y, tau_pt,
                grad_outputs=residual_0)[0]
        # Do summations to convert tau gradient to A/b gradient
        tau_dots_vsums = torch.sum(tau_dots, axis=(2,))
        tau_dots_usums = torch.sum(tau_dots, axis=(1,))
        # Contract
        m_contractions = torch.einsum('abc,b -> ac', tau_dots_vsums,
                shift_scale[0,1]*m_vec)
        n_contractions = torch.einsum('abc,b -> ac', tau_dots_usums,
                shift_scale[0,0]*n_vec)
        # Assemble into A gradient
        grad_A_tau = torch.cat((
            n_contractions[..., None],
            m_contractions[..., None]), -1)
        # b grad
        grad_b = torch.sum(tau_dots_usums, axis=(1,)) * shift_scale
        # Create gradient-wrt-filter term
        hadamard = torch.sum(G_adap[:, None, ...] * residual_1, 1)
        paint = (torch.transpose(hadamard, -1,
            -2).reshape(batch_sz,-1)[...,None] * grid[None,...])
        scale_matrix = torch.einsum('bij,ik->bjk', paint, grid)
        scale_term0 = torch.sum(hadamard, axis=(-1,-2))
        # Get the gradients
        conj_scale = (torch.bmm(torch.bmm(Sigmainv, scale_matrix), Sigmainv) -
                scale_term0[:,None,None] * Sigmainv)
        grad_A_filter = (sigma0**2 *
                torch.bmm(torch.bmm(torch.transpose(A,-1,-2), conj_scale),
                    AAt) + scale_term0[:,None,None] *
                torch.transpose(A,-1,-2))

        # Accumulate
        grad_A = grad_A_tau + grad_A_filter

        # Rescale the gradients (inverse parameterization)
        grad_b_rs = -torch.bmm(torch.transpose(Ainv, -1, -2),
                grad_b[...,None])
        grad_A_rs = -(torch.bmm(torch.bmm(torch.transpose(Ainv, -1, -2), grad_A),
            torch.transpose(Ainv, -1, -2)) + grad_b_rs *
            torch.transpose(Ainv_b[..., None], -2, -1))

        ## DEBUG: Check finite differences
        #dir_A = torch.randn((batch_sz, 2, 2), device=dev, dtype=precision())
        #dir_b = torch.randn((batch_sz, 2), device=dev, dtype=precision())
        #dirderivs_A = torch.sum(dir_A * grad_A, (-1,-2))
        #dirderivs_b = torch.sum(dir_b * grad_b, (-1))
        #Ntrial = 20
        #t = torch.logspace(-10, -1, Ntrial)
        #finite_diffs = torch.zeros((Ntrial,batch_sz), device=dev,
        #        dtype=precision())
        #for idx_trial in range(Ntrial):
        #    tval = t[idx_trial]
        #    #resample_tmp = get_affine_grid_from_basis(A, b + tval * dir_b,
        #    #        basis_u, basis_v, basis_shift, shift_scale)
        #    resample_tmp = get_affine_grid_from_basis(A + tval*dir_A, b,
        #            basis_u, basis_v, basis_shift, shift_scale)
        #    Y_tmp = resample_chunked(Y, resample_tmp, chunk_sz)
        #    G_tmp = gaussian_cov_adaptive(A+tval*dir_A, M=2*G_tail+1,
        #            sigma=sigma, sigma0=sigma0)
        #    y_conv_tmp = conv2d(Y_tmp.detach(),
        #            G_tmp[None,...].expand(C,-1,-1,-1), padding=G_tail,
        #            groups=C)
        #    fres_tmp = (y_conv_tmp - X_1conv)
        #    fwres_tmp = mask[None,...] * fres_tmp
        #    cost = 0.5/C * torch.sum(fwres_tmp**2, axis=(-1,-2,-3))
        #    cost0 = error[:, idx]
        #    finite_diffs[idx_trial,:] = 1/tval * (cost - cost0)
        #print(f'finite diffs estimate: {finite_diffs[0,:]}')
        #print(f'directional deriv: {dirderivs_A}')

        # Gradient steps
        A -= step_A * grad_A_rs
        b -= step_b * grad_b_rs[...,0]
        # Recalculate stuff
        Ainv = torch.linalg.inv(A)
        Ainv_b = torch.bmm(Ainv, b[...,None])[..., 0]
        tau_pt = get_affine_grid_from_basis_xy(Ainv, -Ainv_b, basis_u, basis_v,
                basis_shift, shift_scale)

        if record_process:
            A_list[idx+1,:,:] = A
            b_list[idx+1,:] = b
            
            cu = cur_Y.shape[2] // 2
            cv = cur_Y.shape[3] // 2
            
            cur_Y_ctr = cur_Y[:, :, cu-roi_u//2:cu+roi_u//2+1, 
                                    cv-roi_v//2:cv+roi_v//2+1]
            Rvals[idx] = torch.sum(cur_Y_ctr[0] * X) / \
                    torch.sqrt(torch.sum(cur_Y_ctr[0]**2) * torch.sum(X**2))

    # Cleanup: last-iter operations
    cur_Y = resample_chunked(Y, tau_pt, chunk_sz)
    y_conv = conv2d(cur_Y.detach(), G_adap[None,...].expand(C,-1,-1,-1),
            padding=G_tail, groups=C)
    f_res = (y_conv - X_1conv)
    fw_res = mask[None,...] * f_res
    error[accept_idxs, max_iter-1] = 0.5/C * torch.sum(fw_res**2, axis=(-1,-2,-3))
    # fill in the cost on the reject indices
    reject_mask = torch.ones((batch_sz,), device=dev, dtype=bool)
    reject_mask[accept_idxs] = False
    reject_idxs = torch.arange(batch_sz)[reject_mask]
    error[reject_idxs, :] = error[accept_idxs,0].max()
    # default settings for the rejected final transformation params
    b_out = torch.zeros((batch_sz, 2), device=dev, dtype=precision())
    A_out = torch.eye(2, device=dev,
            dtype=precision())[None,...].expand(batch_sz, -1, -1)
    b_out[accept_idxs, :] = b
    A_out[accept_idxs, :] = A
    # For comparison, get a sigma0-smoothed and matching-padded motif
    if external_smoothing == False: 
        cur_X = conv2d(X[None,...], G_scene_pt, padding=2*G_scene_tail +
                2*G_tail, groups=C)
    else:
        # Just add padding to compare
        cur_X = pad(X[None, ...], (2*G_tail,)*4)
        
    if record_process:
        cu = cur_Y.shape[2] // 2
        cv = cur_Y.shape[3] // 2

        cur_Y_ctr = cur_Y[:, :, cu-roi_u//2:cu+roi_u//2+1, 
                                cv-roi_v//2:cv+roi_v//2+1]
        Rvals[-1] = torch.sum(cur_Y_ctr[0] * X) / \
                    torch.sqrt(torch.sum(cur_Y_ctr[0]**2) * torch.sum(X**2))

    # Done
    if record_process:
        return A_out, b_out, error, cur_Y, cur_X, A_list, b_list, Rvals
    else:
        return A_out, b_out, error, cur_Y, cur_X
