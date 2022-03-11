#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions related to data generation.

Module containing functions for loading/creating test image data.
"""

# imports
import numpy as np
import numpy.linalg as npla
import scipy as sp
import matplotlib.pyplot as plt

def crab_beach(b, A=np.eye(2)):
    """Create image of the crab on the beach at specified location

    translation parameter b is interpreted in inverse form

    """

    from matplotlib.image import imread
    from registration_np import (affine_to_vf, image_interpolation_bicubic)

    # Load the crab 
    img = imread('crab/crab.png')
    crab = img[..., :3]
    mask = img[..., 3]

    # Load the beach bg
    beach_bg = imread('data/beach_bg.jpg')
    beach_bg = 1/255 * beach_bg.astype('float64')

    M, N, C = crab.shape
    Ms, Ns, C = beach_bg.shape

    # Get specified embed location
    center = np.array([(M-1)/2, (N-1)/2])
    corr = np.dot(np.eye(2) - A, center)
    (tau_u, tau_v) = affine_to_vf(A, -A@b + corr, Ms, Ns)
    crab_emb = image_interpolation_bicubic(crab, tau_u, tau_v)
    mask_emb = np.round(image_interpolation_bicubic(mask[...,None], tau_u, tau_v))

    scene = mask_emb * crab_emb + (1.0 - mask_emb) * beach_bg

    return scene
