#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions related to the crab template, and its motifs.

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

def crab(prefix=""):
    """Load the black-background crab motif
    
    optional argument side says to load either the left claw or the right claw
    (relative to our viewpoint -- left claw = left side of image of crab)

    """

    from matplotlib.image import imread

    crab = imread(prefix + '/crab.png')

    crab =  crab.astype('float64')

    motif = crab[...,:-1]
    mask = crab[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def claw(prefix="", side="left"):
    """Load the black-background crab claw motif
    
    optional argument side says to load either the left claw or the right claw
    (relative to our viewpoint -- left claw = left side of image of crab)

    """

    from matplotlib.image import imread

    if side == "left":
        claw = imread(prefix + '/left_claw.png')
    else:
        claw = imread(prefix + '/right_claw.png')

    claw =  claw.astype('float64')

    motif = claw[...,:-1]
    mask = claw[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def eye(prefix="", side="left"):
    """Load the black-background crab eye motif
    
    optional argument side says to load either the left eye or the right eye
    (relative to our viewpoint -- left eye = left side of image of crab)

    """

    from matplotlib.image import imread

    if side == "left":
        eye = imread(prefix + '/left_eye.png')
    else:
        eye = imread(prefix + '/right_eye.png')

    eye =  eye.astype('float64')

    motif = eye[...,:-1]
    mask = eye[...,-1]

    dev = device()
    motif = torch.moveaxis(tensor(motif, device=dev, dtype=precision()), -1,
            0)
    mask = tensor(mask, device=dev, dtype=precision())[None, ...]

    return motif, mask

def hierarchy(base_path=""):
    """Returns the hierarchy graph for the crab template.

    Used for motif extraction / detection in motifs.py and detection.py.

    NOTES (amateur stuff):
    1. General graph info is accessed at G.graph (a dict)
    2. 

    Inputs:
    ------------
    base_path: str
        Optional prefix to add to all file names. Used when calling these files
        from another directory (e.g. a notebook in the experiments directory)

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
    prec = precision()

    # Just use two dicts. One for textured, one for spike. (Tune later.)
    # Dict for textured eye motifs
    eye_dict = {}
    eye_dict['image_type'] = 'textured'
    eye_dict['step'] = tensor(2e-4, device=dev, dtype=precision())
    eye_dict['sigma'] = 3
    eye_dict['sigma_scene'] = 1.5
    eye_dict['max_iter'] = 64
    eye_dict['stride_u'] = 20
    eye_dict['stride_v'] = 20
    eye_dict['nu'] = 10
    # left eye
    left_eye_dict = eye_dict.copy()
    left_eye_dict['thresh'] = 10
    left_eye_dict['articulation_pt'] = torch.tensor(((82,4),), device=dev,
            dtype=prec)
    right_eye_dict = eye_dict.copy()
    right_eye_dict['thresh'] = 10
    right_eye_dict['articulation_pt'] = torch.tensor(((81,4),), device=dev,
            dtype=prec)
    # Dict for textured claw motifs
    # Needs a smaller step for some reason...
    claw_dict = eye_dict.copy()
    left_claw_dict = claw_dict.copy()
    left_claw_dict['thresh'] = 10
    left_claw_dict['step'] /= 2
    left_claw_dict['articulation_pt'] = center=torch.tensor(((51,38),),
            device=dev, dtype=prec)
    right_claw_dict = claw_dict.copy()
    right_claw_dict['thresh'] = 10
    right_claw_dict['articulation_pt'] = center=torch.tensor(((45,3),),
            device=dev, dtype=prec)
    # right_claw_dict['step'] /= 2
    # Dict for spike motifs
    # This one has an aggressive step size...
    spike_dict = {}
    spike_dict['image_type'] = 'spike'
    spike_dict['step'] = tensor(2e-1, device=dev, dtype=precision())
    spike_dict['sigma'] = 20
    spike_dict['sigma_scene'] = 3
    spike_dict['max_iter'] = 32
    spike_dict['stride_u'] = 20
    spike_dict['stride_v'] = 20
    spike_dict['nu'] = 2.5e5
    spike_dict['thresh'] = 1e-5
    # Dict for top, which needs to be different...
    crab_dict = spike_dict.copy()
    crab_dict['max_iter'] = 32
    #crab_dict['step'] *= 2
    crab_dict['thresh'] = 2e-5

    # TODO: Hardcode the articulation points for each motif for now.
    # Would be better to set these in the eye() and claw() functions.
    # Right now they're also hardcoded in the motion code (in the fig3 notebook)
    G = nx.DiGraph(root=root, cache_fn=base_path + '/crab_hierarchical')
    G.add_node(root, content=crab(prefix=base_path), params=crab_dict.copy())
    G.add_node(child0, content=claw(prefix=base_path, side="left"),
            params=left_claw_dict.copy(),)
    G.add_node(child1, content=claw(prefix=base_path, side="right"),
            params=right_claw_dict.copy(),)
    G.add_node(child2, content=(np.zeros((1,1,1)),)*2, params=spike_dict.copy())
    G.add_node(child3, content=eye(prefix=base_path, side="left"),
            params=left_eye_dict.copy())
    G.add_node(child4, content=eye(prefix=base_path, side="right"),
            params=right_eye_dict.copy(),)
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
