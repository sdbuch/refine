#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import numpy as np
import numpy.linalg as npla
from numpy.random import default_rng
import scipy as sp
import matplotlib.pyplot as plt
import torch
from torch import tensor
from registration_pt import device, precision

"""Code for invariant object detection routines.

These are higher-level than some other modules, e.g. they build on code in
registration.py

To be called externally.

"""

def bfsw_detector_pt(Y, X, mask, image_type='textured', sigma=3,
        sigma_scene=1.5, nu=1, stride_u=16, stride_v=16, max_iter=1024,
        thresh=0, step=tensor(1e0, device=device(), dtype=precision())):
    """Brute Force Sliding Window optimization-based detector

    Search for a motif X in a scene Y by "brute force": given striding
    parameters param_dict['stride_u'], param_dict['stride_v'], run a
    registration algorithm placed at each stride point in Y to obtain a final
    loss and final transformation parameters, and use those data to generate a
    detection occurrence map ("rolloff" controlled by param_dict['nu']) with
    coordinates the same as the input scene Y which determines where and "to
    what intensity" the motif X was detected at each point in the image.  The
    idea being, that when the stride parameters are set sufficiently small
    (i.e. dense searching), we will find all occurrences of the motif; and by
    using registration, we can do the detection in an invariant/potentially
    efficient fashion.

    The "detector" terminology is a slight misnomer, as the main output here is
    an occurrence map -- in other words, detection is left to be done by
    external algorithms (no thresholding/etc is done here other than rolloff
    control via param_dict['nu'] mentioned above).


    Inputs:
    ------------
    Y - (C, M, N) torch tensor
        Input scene, strided over to search for X
    X - (C, m, n) torch tensor
        Input motif, to be detected in the scene Y
    mask - (1, M, N) torch tensor, or None
        Either the mask for the motif, or anything with image_mode='spike' (it
        gets ignored)
    Other parameters: see registration_pt documentation

    Outputs:
    ------------
    spike map, coord map, errors map, transformation map

    """

    import torch
    from registration_pt import (device, reg_l2_rigid)
    from images import gaussian_filter_2d_pt, gaussian_cov

    # Overhead
    dev = device()

    # Main tasks:

    C, M, N = Y.shape
    c, m, n = X.shape

    # 1. Process the strides
    start_u = 0
    stop_u = M
    start_v = 0
    stop_v = N
    start_u_chk = int(np.maximum(0, start_u))
    start_v_chk = int(np.maximum(0, start_v))
    stop_u_chk = int(np.minimum(M - m + 1, stop_u))
    stop_v_chk = int(np.minimum(N - n + 1, stop_v))
    anchors_u = torch.arange(start_u_chk, stop_u_chk, stride_u, device=dev,
            dtype=precision())
    anchors_v = torch.arange(start_v_chk, stop_v_chk, stride_v, device=dev,
            dtype=precision())
    # add the end indices too, to avoid a "near zero bias"...
    if anchors_u[-1] < stop_u_chk-1:
        anchors_u = torch.cat((anchors_u, torch.tensor((stop_u_chk-1,),
            device=dev, dtype=precision())), dim=-1)
    if anchors_v[-1] < stop_v_chk-1:
        anchors_v = torch.cat((anchors_v, torch.tensor((stop_v_chk-1,),
            device=dev, dtype=precision())), dim=-1)

    # 2. Do the registration with batched reg_l2 (create locs)
    locs = torch.cat(
            (torch.kron(anchors_u, torch.ones(len(anchors_v), device=dev,
                dtype=precision()))[:,None],
                torch.kron(torch.ones(len(anchors_u), device=dev,
                    dtype=precision()), anchors_v)[:,None]), dim=-1)
    print(f'Total work to do: {locs.shape[0]} strides, '
            f'at {X.shape[1]} by {X.shape[2]} motif, '
            f'in {Y.shape[1]} by {Y.shape[2]} scene.')
    if image_type == 'textured':
        # Textured motif call: two rounds of multiscale
        phi, b, errors0 = reg_l2_rigid(Y, X, mask, locs, step=step,
                max_iter=max_iter, sigma=sigma, sigma_scene=sigma_scene)
        # Second round multiscale
        phi, b, errors2 = reg_l2_rigid(Y, X, mask, locs, step=step,
                max_iter=256, sigma=0.1, sigma_scene=1e-6, init_data=(phi, b),
                erode=False)
        # Concatenate errors
        errors = torch.cat((errors0, errors2), -1)
    else:
        # spike motif call: don't need multiscale
        phi, b, errors = reg_l2_rigid(Y, X, mask, locs, step=step,
                max_iter=max_iter, sigma=sigma, sigma_scene=sigma_scene,
                image_type='spike', rejection_thresh=0.2)

    # 3. Create the spike map (do this on the GPU)
    ctr = torch.tensor(((m-1)/2, (n-1)/2), device=dev,
            dtype=precision())[None, :]
    spike_locs = torch.min(torch.max(ctr + locs + torch.flip(b, (-1,)),
        torch.tensor((0,0),device=dev,dtype=precision())),
        torch.tensor((M,N),device=dev,dtype=precision()))
    
    spike_map = torch.zeros((1, M, N), device=dev, dtype=precision())
    for idx in range(spike_locs.shape[0]):
        weight = torch.exp(-nu * torch.maximum(torch.zeros((1,), device=dev,
            dtype=precision()), errors[idx, -1] - thresh))
        spike_map = torch.maximum(spike_map, weight * gaussian_cov(M,
            N=N, Sigma=0.5**2*torch.eye(2,device=dev,dtype=precision()),
            offset_u=spike_locs[idx,0], offset_v=spike_locs[idx,1]))

    # 4. Prepare the output struct (need spike map and transformation
    #   parameters, errors for debug). 
    output = {
            'spike_locs': spike_locs,
            'errors': errors,
            'phi': phi,
            'b': b
            }

    return spike_map, output
