#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with image processing routines (in pytorch) 

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
from registration_pt import device, precision

def imagesc(X, ax=None):
    """Call matplotlib's imshow on input X, with automatic greyscale img args.

    Set up for use with pytorch conventions: Expects X is C x M x N tensor

    By default, includes:
    - cmap='gray'
    - vmin, vmax set adaptively (hence imagesc name for fu nction)
    - interpolation='none' (raw pixels)

    """

    img = torch.moveaxis(X, 0, -1).detach().to('cpu').numpy()
    if ax is None:
        fig = plt.figure()
        pos = plt.imshow(img, cmap='gray', vmin=X.min(), vmax=X.max(),
                interpolation='none')
        plt.show()
    else:
        ax.imshow(img, cmap='gray', vmin=X.min(), vmax=X.max(),
                interpolation='none')

def show_false_color(X_in,normalize=False, ax=None, invert_color=False, return_img=False):
    """Plot a false color image of multichannel input image X

    Inputs:
    ---------
    X - M x N x C image/numpy array
        The input scene to plot 
    normalize - bool (default True)
        If evaluates to True, each channel in the false color image is divided
        by its maximum value (to make it more visible)

    Outputs:
    ----------
    none. calls plt.show() inside the function

    """

    import matplotlib.colors as mpc

    X = torch.moveaxis(X_in, 0, -1).detach().to('cpu').numpy() 
    Xfc = np.zeros((X.shape[0],X.shape[1],3))

    allColors = ['red','green','blue','orange','yellow','fuchsia','purple','aqua','crimson','lime']

    for i in range(X.shape[2]):
        curColor = mpc.to_rgba(allColors[i % len(allColors)])
        if invert_color:
            curColor = [1-c for c in curColor]

        Xfc[:,:,0] += curColor[0] * X[:,:,i]
        Xfc[:,:,1] += curColor[1] * X[:,:,i]
        Xfc[:,:,2] += curColor[2] * X[:,:,i]

    if normalize:
        Xfc = Xfc / np.max(Xfc)
        
    if invert_color:
        Xfc = 1 - Xfc
        
    if ax is None:
        fig = plt.figure()
        plt.imshow(Xfc)
        plt.show()
    else:
        ax.imshow(Xfc)
        
    if return_img:
        return Xfc

def gaussian_filter_1d_pt(N, sigma=1, offset=0):
    """Return a 1D gaussian filter with length N and inverse bandwidth sigma

    The filter is normalized to have unity value at DC (following an
    unnormalized fourier transform).

    Use circulant boundary conditions, with a phase shift to center the filter
    at index 0. The filter is generated to have "dsp flip" symmetry (see
    dsp_flip) regardless of parity of N.

    offset denotes an optional offset parameter, designating where the filter
    is centered -- gets taken modulo N

    """

    i = torch.arange(0, N, device=device(), dtype=precision())

    g = torch.exp(-((i - offset + (N-1)/2) % N - (N-1)/2)**2 / 2/ sigma**2)
    return g / torch.sum(g**2)**(0.5)

def gaussian_filter_2d_pt(M, N=None, sigma_u=1, sigma_v=None, offset_u = 0,
        offset_v = 0):
    """Return a 2D gaussian filter  M by N and inverse bandwidth sigma_{M,N}

    See gaussian_filter_1d: this essentially wraps that function.

    offset_u and offset_v denote optional offset parameters, designating where
    the filter is centered -- offset_u gets taken modulo M, and offset_v gets
    taken modulo N

    """

    if N is None:
        N = M
    if sigma_v is None:
        sigma_v = sigma_u

    # The filter is separable: two 1D filters generate it
    gi = gaussian_filter_1d_pt(M, sigma=sigma_u, offset=offset_u)
    gj = gaussian_filter_1d_pt(N, sigma=sigma_v, offset=offset_v)

    # Expand dimensions for outer product...

    return gi[:,None] * gj[None,:]

def gaussian_cov(M, N=None, Sigma=torch.eye(2,device=device(),
    dtype=precision()), offset_u=None, offset_v=None):
    """Return a 2D Gaussian filter with size MxN and covariance Sigma

    This is for use with pytorch, so the filter is "fftshift-centered" (what
    one gets if centering at 0, as in the other routines, then calling fftshift
    on the result)

    """

    if N is None:
        N = M
    if offset_u is None:
        offset_u = M//2
    if offset_v is None:
        offset_v = N//2

    dev = device()

    # Make grid in kronecker style
    m_vec = torch.arange(M, device=dev, dtype=precision()) - offset_u
    n_vec = torch.arange(N, device=dev, dtype=precision()) - offset_v
    grid = torch.cat( (torch.kron(m_vec, torch.ones(len(n_vec), device=dev,
        dtype=precision()))[:,None], torch.kron(torch.ones(len(m_vec),
            device=dev, dtype=precision()), n_vec)[:,None]), dim=-1)

    # Exponent
    exponent = -0.5 * (torch.sum(torch.matmul(grid, torch.linalg.inv(Sigma)) *
        grid, dim=-1) + torch.logdet(Sigma))
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    filter = 0.5/pi * torch.exp(torch.reshape(exponent, (M, N)))

    return filter

def gaussian_cov_adaptive(A, M=10, sigma=10, sigma0=1, N=None):
    """gaussian filter MxN and covariance generated from 2x2 A w/compensation

    Runs in batch mode (pass A as B x 2 x 2 tensor). Could feasibly support
    multi-sigma batch mode but not implementing this for now.

    This is for use with pytorch, so the filter is "fftshift-centered" (what
    one gets if centering at 0, as in the other routines, then calling fftshift
    on the result)

    NOTE: No input checking is performed here. I think things will fail
    (numerically) when any input matrix A is too ill-conditioned (the
    singularity threshold should be sigma0 / sigma, for the smallest
    eigenvalue's absolute value?)

    NOTE: Really requires M and N to be appropriately large given sigma (no
    extra normalization is introduced past the definition of the pdf)

    """

    if N is None:
        N = M
    B = A.shape[0]

    dev = device()

    identity = torch.eye(2,device=dev, dtype=precision())

    AtA = torch.bmm(torch.transpose(A,-1,-2), A)
    AtAinv = torch.linalg.inv(AtA)
    Sigma = (sigma**2 * identity[None,...].expand(B,2,2) - sigma0**2 * AtAinv)
    Sigmainv = torch.linalg.inv(Sigma)

    # Make grid in kronecker style
    m_vec = torch.arange(M, device=dev, dtype=precision()) - M//2
    n_vec = torch.arange(N, device=dev, dtype=precision()) - N//2
    grid = torch.cat( (torch.kron(m_vec, torch.ones(len(n_vec), device=dev,
        dtype=precision()))[:,None], torch.kron(torch.ones(len(m_vec),
            device=dev, dtype=precision()), n_vec)[:,None]), dim=-1)

    # Exponent
    exponent = -0.5 * (torch.sum(torch.bmm(Sigmainv,
        grid.T[None,...].expand(B,-1,-1)) * grid.T[None,...], dim=-2) +
        torch.logdet(Sigma)[...,None])
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    filters = (0.5/pi * torch.exp(torch.reshape(exponent, (B, M, N))) *
            torch.sqrt(torch.det(AtA))[:,None,None])

    return filters

def gaussian_cov_adaptive_xy(A, M=10, sigma=10, sigma0=1, N=None):
    """gaussian filter MxN and covariance generated from 2x2 A w/compensation

    Runs in batch mode (pass A as B x 2 x 2 tensor). Could feasibly support
    multi-sigma batch mode but not implementing this for now.

    This is for use with pytorch, so the filter is "fftshift-centered" (what
    one gets if centering at 0, as in the other routines, then calling fftshift
    on the result)

    NOTE: No input checking is performed here. I think things will fail
    (numerically) when any input matrix A is too ill-conditioned (the
    singularity threshold should be sigma0 / sigma, for the smallest
    eigenvalue's absolute value?)

    NOTE: Really requires M and N to be appropriately large given sigma (no
    extra normalization is introduced past the definition of the pdf)
    """

    if N is None:
        N = M
    B = A.shape[0]

    dev = device()

    identity = torch.eye(2,device=dev, dtype=precision())

    AtA = torch.bmm(torch.transpose(A,-1,-2), A)
    AtAinv = torch.linalg.inv(AtA)
    Sigma = (sigma**2 * identity[None,...].expand(B,2,2) - sigma0**2 * AtAinv)
    Sigmainv = torch.linalg.inv(Sigma)

    # Make grid in kronecker style
    m_vec = torch.arange(M, device=dev, dtype=precision()) - M//2
    n_vec = torch.arange(N, device=dev, dtype=precision()) - N//2
    grid = torch.cat( (torch.kron(n_vec, torch.ones(len(m_vec), device=dev,
        dtype=precision()))[:,None], torch.kron(torch.ones(len(n_vec),
            device=dev, dtype=precision()), m_vec)[:,None]), dim=-1)

    # Exponent
    exponent = -0.5 * (torch.sum(torch.bmm(Sigmainv,
        grid.T[None,...].expand(B,-1,-1)) * grid.T[None,...], dim=-2) +
        torch.logdet(Sigma)[...,None])
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    filters = (0.5/pi * torch.exp(torch.reshape(exponent, (B, N, M))) *
            torch.sqrt(torch.det(AtA))[:,None,None])

    return torch.transpose(filters, -1,-2)

def gaussian_cov_adaptive_dot(A, V, sigma=10, sigma0=1):
    """gaussian filter MxN and covariance generated from 2x2 A w/compensation

    Runs in batch mode (pass A as B x 2 x 2 tensor). Could feasibly support
    multi-sigma batch mode but not implementing this for now.

    Return the derivative (JVP). This is performed if V is
    passed: V should be B x C x M x N shaped (i.e. handle scene cropping outside
    of this function, and tolerate multi-channel setting)

    This is for use with pytorch, so the filter is "fftshift-centered" (what
    one gets if centering at 0, as in the other routines, then calling fftshift
    on the result)

    NOTE: No input checking is performed here. I think things will fail
    (numerically) when any input matrix A is too ill-conditioned (the
    singularity threshold should be sigma0 / sigma, for the smallest
    eigenvalue's absolute value?)

    NOTE: Really requires M and N to be appropriately large given sigma (no
    extra normalization is introduced past the definition of the pdf)

    """

    B, C, M, N = V.shape

    dev = device()

    identity = torch.eye(2,device=dev, dtype=precision())

    AtA = torch.bmm(torch.transpose(A,-1,-2), A)
    AtAinv = torch.linalg.inv(AtA)
    Sigma = (sigma**2 * identity[None,...].expand(B,2,2) - sigma0**2 * AtAinv)
    Sigmainv = torch.linalg.inv(Sigma)

    # Make grid in kronecker style
    m_vec = torch.arange(M, device=dev, dtype=precision()) - M//2
    n_vec = torch.arange(N, device=dev, dtype=precision()) - N//2
    grid = torch.cat( (torch.kron(m_vec, torch.ones(len(n_vec), device=dev,
        dtype=precision()))[:,None], torch.kron(torch.ones(len(m_vec),
            device=dev, dtype=precision()), n_vec)[:,None]), dim=-1)

    # Exponent
    exponent = -0.5 * (torch.sum(torch.bmm(Sigmainv,
        grid.T[None,...].expand(B,-1,-1)) * grid.T[None,...], dim=-2) +
        torch.logdet(Sigma)[...,None])
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    filters = (0.5/pi * torch.exp(torch.reshape(exponent, (B, M, N))) *
            torch.sqrt(torch.det(AtA))[:,None,None])

    # Do derivative stuff
    # Nesting it because we'll reuse some things computed above
    Ainv = torch.linalg.inv(A)
    # Calculate the scale matrix
    hadamard = torch.sum(filters[:, None, ...] * V, 1)
    paint = hadamard.reshape(B,-1)[...,None] * grid[None,...]
    scale_matrix = torch.einsum('bij,ik->bjk', paint, grid)
    scale_term = torch.sum(hadamard, axis=(-1,-2))
    # Get the gradients
    conj_scale = (torch.bmm(torch.bmm(Sigmainv, scale_matrix), Sigmainv) -
            scale_term[:,None,None] * Sigmainv)
    grads = (sigma0**2 * torch.bmm(torch.bmm(torch.transpose(Ainv,-1,-2),
        conj_scale), AtAinv) +
        scale_term[:,None,None]*torch.transpose(Ainv,-1,-2))

    return grads

def gaussian_cov_adaptive_dot_xy(A, V, sigma=10, sigma0=1):
    """gaussian filter MxN and covariance generated from 2x2 A w/compensation

    Runs in batch mode (pass A as B x 2 x 2 tensor). Could feasibly support
    multi-sigma batch mode but not implementing this for now.

    Return the derivative (JVP). This is performed if V is
    passed: V should be B x C x M x N shaped (i.e. handle scene cropping outside
    of this function, and tolerate multi-channel setting)

    This is for use with pytorch, so the filter is "fftshift-centered" (what
    one gets if centering at 0, as in the other routines, then calling fftshift
    on the result)

    NOTE: No input checking is performed here. I think things will fail
    (numerically) when any input matrix A is too ill-conditioned (the
    singularity threshold should be sigma0 / sigma, for the smallest
    eigenvalue's absolute value?)

    NOTE: Really requires M and N to be appropriately large given sigma (no
    extra normalization is introduced past the definition of the pdf)

    """

    B, C, M, N = V.shape

    dev = device()

    identity = torch.eye(2,device=dev, dtype=precision())

    AtA = torch.bmm(torch.transpose(A,-1,-2), A)
    AtAinv = torch.linalg.inv(AtA)
    Sigma = (sigma**2 * identity[None,...].expand(B,2,2) - sigma0**2 * AtAinv)
    Sigmainv = torch.linalg.inv(Sigma)

    # Make grid in kronecker style
    m_vec = torch.arange(M, device=dev, dtype=precision()) - M//2
    n_vec = torch.arange(N, device=dev, dtype=precision()) - N//2
    grid = torch.cat( (torch.kron(n_vec, torch.ones(len(m_vec), device=dev,
        dtype=precision()))[:,None], torch.kron(torch.ones(len(n_vec),
            device=dev, dtype=precision()), m_vec)[:,None]), dim=-1)

    # Exponent
    exponent = -0.5 * (torch.sum(torch.bmm(Sigmainv,
        grid.T[None,...].expand(B,-1,-1)) * grid.T[None,...], dim=-2) +
        torch.logdet(Sigma)[...,None])
    pi = torch.tensor(np.pi,device=dev,dtype=precision())
    filters = (0.5/pi * torch.exp(torch.reshape(exponent, (B, N, M))) *
            torch.sqrt(torch.det(AtA))[:,None,None])
    filters = torch.transpose(filters, -1, -2)

    # Do derivative stuff
    # Nesting it because we'll reuse some things computed above
    Ainv = torch.linalg.inv(A)
    # Calculate the scale matrix
    hadamard = torch.sum(filters[:, None, ...] * V, 1)
    paint = (torch.transpose(hadamard,-1,-2).reshape(B,-1)[...,None] *
            grid[None,...])
    scale_matrix = torch.einsum('bij,ik->bjk', paint, grid)
    scale_term = torch.sum(hadamard, axis=(-1,-2))
    # Get the gradients
    conj_scale = (torch.bmm(torch.bmm(Sigmainv, scale_matrix), Sigmainv) -
            scale_term[:,None,None] * Sigmainv)
    grads = (sigma0**2 * torch.bmm(torch.bmm(torch.transpose(Ainv,-1,-2),
        conj_scale), AtAinv) +
        scale_term[:,None,None]*torch.transpose(Ainv,-1,-2))

    return grads

def mask_to_bbox_pt(mask, thresh=0):
    """Convert a mask (may be {0,1}-valued image) to smallest enclosing rectangle

    Converts an object mask to its corresponding bounding box -- the
    parameters corresponding to its smallest containing rectangle.

    Inputs:
        mask : (1, M, N) numpy array
        Assumed to have values only 0 or 1.

    Outputs:
        bbox_left : int
        bbox_right : int
        bbox_top : int
        bbox_bot : int

        The four index parameters that define the bounding box, such that the
        box corresponds to the sliced image 
        mask[bbox_top:bbox_bot, bbox_left:bbox_right].
    
    Caveats:
        The "smallest enclosing rectangle" for the zero input is treated
        manually as the zero output

    """

    C, M, N = mask.shape
    
    slice_u = torch.sum(mask, (0,2))
    slice_v = torch.sum(mask, (0,1))
    u_supp = torch.where(slice_u > thresh)[0]
    v_supp = torch.where(slice_v > thresh)[0]

    if len(u_supp) > 0 and len(v_supp) > 0:
        bb_left = v_supp[0]
        bb_right = v_supp[-1]
        bb_top = u_supp[0]
        bb_bot = u_supp[-1]

    else:
        # "null parameters" to define an empty bounding box
        bb_left = N
        bb_right = N-1
        bb_top = M
        bb_bot = M-1

    return bb_left, bb_right+1, bb_top, bb_bot+1

def get_affine_grid_basis(M, N, extent_M, extent_N, locs=None, ctrs=None):
    """Get a matrix basis for a grid to use with pytorch grid_sample

    This function returns the matrix bases used to generate an affine grid. It's
    separated this way since in registration methods, we need to repeatedly
    update the grid we sample on, so it makes sense to keep the possibly large
    grid around rather than repeatedly recompute it. Call with
    get_affine_grid_from_basis for easy generation of the grid (or combine with
    get_affine_grid, which just wraps both for one-off grid generation).

    Supports batched computation (wrt loc, center)

    What defines the grid:
        Image size (M, N)
        Grid starting coordinate loc
        Grid extent
        Grid center

    In normal cases, we like our grids to be symmetric about the center. This
    means that loc and center are coupled -- in particular we have ctrs = locs +
    (extent-1)/2. Here, however, it's possible to specify a general center.

    In practice, the grid corresponds to a region in a scene where we're trying
    to match a specific motif, with smoothing. This means that in practice, the
    grid extent is coupled to the size of the (smoothed) motif.

    """

    dev = device()

    # Support various processing modes:
    # 1. Only locs is passed. Infer ctrs from locs
    # 2. Only ctrs is passed. Infer locs from ctrs
    # 3. Neither is passed.     
    # 4. Both are passed. Go with the flow and do whatever is specified
    if locs is None and ctrs is None:
        # Create the grid for the entire image plane, centered at its center
        locs = torch.tensor((0,0), device=dev, dtype=precision())[None,...]
        ctrs = torch.tensor((((M-1)/2,(N-1)/2),), device=dev, dtype=precision())
    elif ctrs is None:
        # Infer ctrs from locs
        extent = torch.tensor(((extent_M, extent_N),), device=dev,
                dtype=precision())
        ctrs = locs + (extent - 1)/2
    elif locs is None:
        # Infer locs from ctrs
        extent = torch.tensor(((extent_M, extent_N),), device=dev,
                dtype=precision())
        locs = ctrs - (extent - 1)/2
    else:
        # Sit tight
        pass

    # Batched creation of the grid basis
    m_vec = torch.arange(extent_M, device=dev, dtype=precision())
    m_vec = m_vec[None, ...] + locs[:, 0][:, None] - ctrs[:, 0][:,None]
    n_vec = torch.arange(extent_N, device=dev, dtype=precision())
    n_vec = n_vec[None, ...] + locs[:, 1][:, None] - ctrs[:, 1][:,None]
    tau_basis_u = 2/(M-1) * torch.einsum('abc,cd -> abd', m_vec[...,None],
            torch.ones((1,extent_N),device=dev, dtype=precision()))[..., None]
    tau_basis_v = 2/(N-1) * torch.einsum('abc,cd -> adb', n_vec[...,None],
            torch.ones((1,extent_M),device=dev, dtype=precision()))[..., None]
    image_ctr = torch.tensor((((M-1)/2, (N-1)/2),), device=dev,
                dtype=precision())
    tau_basis_shift = ((ctrs - image_ctr) / image_ctr)[:, None, None, :]
    tau_basis_shift = torch.flip(tau_basis_shift, (-1,))
    tau_basis_scale = 1/image_ctr
    tau_basis_scale = torch.flip(tau_basis_scale, (-1,))

    return tau_basis_u, tau_basis_v, tau_basis_shift, tau_basis_scale

def get_affine_grid_from_basis(A, b, basis_u, basis_v, basis_shift,
        shift_scale):
    """Create a grid from grid bases, using provided matrices A, b (u,v coord)

    This function outputs in (x, y) coordinates rather than (u, v) coordinates,
    to be compatible with grid_sample. 
    But A should still be passed acting on (u, v) coordinates (permuting is
    handled here), whereas b is interpreted as (x,y) coords already

    """

    tau_pt = (torch.cat((A[:,1,0,None,None] * basis_u + A[:,1,1,None,None] *
        basis_v, A[:,0,0,None,None] * basis_u + A[:,0,1,None,None] * basis_v),
        axis=-1) + basis_shift + (b * shift_scale)[:, None, None, :])

    return tau_pt

def get_affine_grid_from_basis_xy(A, b, basis_u, basis_v, basis_shift,
        shift_scale):
    """Create a grid from grid bases, using provided matrices A, b (x,y coord)

    This function outputs and inputs with (x, y) coordinates rather than (u, v)
    coordinates, to be compatible with grid_sample. 
    To convert a (u,v) A matrix to an (x,y) A matrix, swap the diagonal elements
    and off-diagonal elements.


    """

    # Could be faster to do this with a BMM operation (kronecker style)
    # plus reshaping... maybe not though (lots of size-2 vector dot products)
    tau_pt = (torch.cat((A[:,0,1,None,None] * basis_u + A[:,0,0,None,None] *
        basis_v, A[:,1,1,None,None] * basis_u + A[:,1,0,None,None] * basis_v),
        axis=-1) + basis_shift + (b * shift_scale)[:, None, None, :])

    return tau_pt

def get_affine_grid(A, b, M, N, extent_M, extent_N, locs=None, ctrs=None):
    """Create grid for affine transform. Wrapper for convenience"""

    basis_u, basis_v, basis_shift, shift_scale = get_affine_grid_basis(M, N,
            extent_M, extent_N, locs=locs, ctrs=ctrs)
    return get_affine_grid_from_basis(A, b, basis_u, basis_v, basis_shift,
            shift_scale)
