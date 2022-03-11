#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions related to the crab template, and its motifs.

"""

# imports
import numpy as np
import numpy.linalg as npla
from numpy.random import default_rng
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from time import sleep
import torch
from torch import tensor
from registration_pt import device, precision

def crab():
    """Load the black-background crab motif
    
    optional argument side says to load either the left claw or the right claw
    (relative to our viewpoint -- left claw = left side of image of crab)

    """

    from matplotlib.image import imread

    crab = imread('../data/crab.png')

    crab =  crab.astype('float64')

    motif = crab[...,:-1]
    mask = crab[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def claw(side="left"):
    """Load the black-background crab claw motif
    
    optional argument side says to load either the left claw or the right claw
    (relative to our viewpoint -- left claw = left side of image of crab)

    """

    from matplotlib.image import imread

    if side == "left":
        claw = imread('../data/left_claw.png')
    else:
        claw = imread('../data/right_claw.png')

    claw =  claw.astype('float64')

    motif = claw[...,:-1]
    mask = claw[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def eye(side="left"):
    """Load the black-background crab eye motif
    
    optional argument side says to load either the left eye or the right eye
    (relative to our viewpoint -- left eye = left side of image of crab)

    """

    from matplotlib.image import imread

    if side == "left":
        eye = imread('../data/left_eye.png')
    else:
        eye = imread('../data/right_eye.png')

    eye =  eye.astype('float64')

    motif = eye[...,:-1]
    mask = eye[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def eye_pair():
    """Load the black-background crab eye pair motif

    """

    from matplotlib.image import imread

    eye_pair = imread('../data/eye_pair.png')

    eye_pair =  eye_pair.astype('float64')

    motif = eye_pair[...,:-1]
    mask = eye_pair[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def hierarchy():
    """Returns the hierarchy graph for the crab template.

    Used for motif extraction / detection in motifs.py and detection.py.

    NOTES (amateur stuff):
    1. General graph info is accessed at G.graph (a dict)
    2. 

    Inputs:
    ------------
    none

    Outputs:
    ----------
    networkx DiGraph (edges denote 'downward' connections in the hierarchy) for
    the object with certain properties:
    - G.graph has a key 'root', which gives the name of the root node (the top
      level motif)
    - G.graph has a key 'cache_fn', which is the file location (relative path)
      at which the graph data will be cached after runs of motif
      extraction/detection algorithms
    - Each node in G has a key 'content', which contains a (2,) tuple with
      first element the motif's image, and second element its mask

    """

    import networkx as nx

    # Graph node identifiers.
    root = "crab"
    child0 = "claw_left"
    child1 = "claw_right"
    child2 = "eye_pair"
    child3 = "eye_left"
    child4 = "eye_right"

    # Construct graph
    # First: nodes
    # Basic content: is the motif/mask getter
    # We will also set up the detection parameter dictionaries here, although
    # this could be done elsewhere.
    dev = device()

    # Just use two dicts. One for textured, one for spike. (Tune later.)
    # Dict for textured eye motifs
    eye_dict = {}
    eye_dict['image_type'] = 'textured'
    eye_dict['step'] = tensor(1e-1, device=dev, dtype=precision())
    eye_dict['sigma'] = 3
    eye_dict['sigma_scene'] = 1.5
    eye_dict['max_iter'] = 1024
    eye_dict['stride_u'] = 20
    eye_dict['stride_v'] = 20
    eye_dict['nu'] = 1
    # left eye
    left_eye_dict = eye_dict.copy()
    left_eye_dict['thresh'] = 58
    right_eye_dict = eye_dict.copy()
    right_eye_dict['thresh'] = 40
    # Dict for textured claw motifs
    # Needs a smaller step for some reason...
    claw_dict = eye_dict.copy()
    left_claw_dict = claw_dict.copy()
    left_claw_dict['thresh'] = 80
    right_claw_dict = claw_dict.copy()
    right_claw_dict['thresh'] = 85
    # Dict for spike motifs
    # This one has an aggressive step size...
    spike_dict = {}
    spike_dict['image_type'] = 'spike'
    spike_dict['step'] = tensor(2e3, device=dev, dtype=precision())
    spike_dict['sigma'] = 10
    spike_dict['sigma_scene'] = 3
    spike_dict['max_iter'] = 1024
    spike_dict['stride_u'] = 20
    spike_dict['stride_v'] = 20
    spike_dict['nu'] = 2.5e5
    spike_dict['thresh'] = 1e-4
    # Dict for top, which needs to be different...
    crab_dict = spike_dict.copy()
    #crab_dict['max_iter'] = 30
    #crab_dict['step'] = 5e2
    crab_dict['thresh'] = 2e-3

    G = nx.DiGraph(root=root, cache_fn='../data/crab_hierarchical')
    G.add_node(root, content=crab(), params=crab_dict.copy())
    G.add_node(child0, content=claw(side="left"), params=left_claw_dict.copy())
    G.add_node(child1, content=claw(side="right"), params=right_claw_dict.copy())
    G.add_node(child2, content=eye_pair(), params=spike_dict.copy())
    G.add_node(child3, content=eye(side="left"), params=left_eye_dict.copy())
    G.add_node(child4, content=eye(side="right"), params=right_eye_dict.copy())
    # Next: edges
    # Top-down starting from motif...
    # Edges don't have properties at init. Add properties to edges when
    # generating composite (spike map) motifs...
    G.add_edge(root, child0)
    G.add_edge(root, child1)
    G.add_edge(root, child2)
    G.add_edge(child2, child3)
    G.add_edge(child2, child4)

    return G