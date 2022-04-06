#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code related to hierarchical motif extraction for invariant detection.

Builds off of functions from detection.py for the extraction. But also wraps a
lot of data structures type stuff.

"""

# imports
import numpy as np
import numpy.linalg as npla
from numpy.random import default_rng
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import tensor
from registration_pt import device, precision
from pdb import set_trace

def extract_hierarchical(G, cal_list=None, suppression_amt=1/20):
    """Calibrate extraction parameters while extracting motifs from hierarchy G

    This function takes as input a basic networkx graph structure for a
    hierarchical object (see crab.py hierarchy function for format example). It
    walks this graph from root to leaves, generating the occurrence maps (spike
    maps) for each motif in the hierarchy along the way, as they will be used
    for detection in novel scenes.  The spike maps are generated according to
    the parameter dict attached to each node of G. 

    *It is assumed* that the extraction corresponds to a certain "easy"
    detection problem here -- namely that the motifs being detected are ripped
    exactly from the base object image (at root of G) and masked, so that in
    reality it is enough to simply register translationally for each motif.

    The work done is similar to detection.bfsw_basin_translation. The flow is
    something like the following:
    - Traverse the graph "upwards" starting from leaves.
    - Update the occurrence maps attached to each node with the result of
      extraction-by-strided-detection
    - Move up to the next level in the traversal. Motifs are now the previous
      level's occurrence maps, and registration_l2_spike is used instead of
      bbg.
    - Continue traversing until done. (stop at top node)

    Inputs:
    -----------
    G - networkx graph object
        Hierarchy for the object to extract motifs from. 
        For our purposes, a hierarchy is a rooted directed acyclic graph (the
        root corresponds to the top-level "object"), with every node reachable
        from the root.
        Needs to have parameter dicts set for each extraction node set
        externally.  See e.g. crab.hierarchy


    Outputs:
    -----------
    none; outputs are written directly to the input graph G.
    Modified fields in the graph G:
        Spike map for each edge; feature map for each node; relevant
        param dicts to each edge (registration) / whole graph (striding);
        error maps / transformation maps for each stride pt (per-edge)

    """

    from detection import bfsw_detector_pt
    from images import mask_to_bbox_pt

    enumeration = enumerate_nodes_for_detection(G)
    Y = G.nodes[G.graph['root']]['content'][0]
    dev = device()

    if cal_list is None:
        cal_list = [motif for motif in G.nodes]

    for depth_idx in range(len(enumeration)):
        for motif_idx in range(len(enumeration[depth_idx])):
            motif = enumeration[depth_idx][motif_idx]
            if cal_list.count(motif) > 0:
                print(f'Processing motif {motif}.')
            else:
                print(f'Motif {motif} not in cal list: skipping.')
                continue

            # Check: is this motif a leaf?
            is_leaf = len(G[motif]) == 0

            # Set up the registration targets
            if is_leaf:
                # Setup the motif for leaf case
                X, mask = G.nodes[motif]['content']
                scene = Y
            else:
                # Setup the motif for composite case
                num_chan = len(G[motif])
                occ_map = torch.zeros((num_chan, Y.shape[1], Y.shape[2]),
                        device=dev, dtype=precision())
                idx = 0
                max_u = -np.inf
                max_v = -np.inf
                for edge in G[motif]:
                    occ_map[idx, ...] = torch.clone(
                            G[motif][edge]['occurrence'][0, ...])
                    if G.nodes[edge]['content'][0].shape[1] > max_u:
                        max_u = G.nodes[edge]['content'][0].shape[1]
                    if G.nodes[edge]['content'][0].shape[2] > max_v:
                        max_v = G.nodes[edge]['content'][0].shape[2]
                    idx += 1
                # Convert the occurrence map to a motif
                occ_mask = tensor(torch.sum(occ_map,0)[None,...] >
                        suppression_amt, device=dev, dtype=precision())
                bb_l, bb_r, bb_t, bb_b = mask_to_bbox_pt(occ_mask)

                # Calculate padding to account for the size of the motif
                pad_u = 0
                pad_v = 0
                # Padding for minimum size
                # if bb_b - bb_t < max_u:
                #     pad_u = (max_u - (bb_b - bb_t))
                # if bb_r - bb_l < max_v:
                #     pad_v = (max_v - (bb_r - bb_l))
                # if pad_u % 2 == 1:
                #     pad_u = pad_u - 1
                # if pad_v % 2 == 1:
                #     pad_v = pad_v - 1
                bb_t = bb_t - pad_u // 2
                bb_b = bb_b + pad_u // 2
                bb_l = bb_l - pad_v // 2
                bb_r = bb_r + pad_v // 2

                # Set the spike map
                G.nodes[motif]['occ_map'] = torch.clone(occ_map[:, bb_t:bb_b,
                    bb_l:bb_r])
                X = G.nodes[motif]['occ_map']
                scene = occ_map
                mask = None
                # Write it to the graph as this node's content too
                G.nodes[motif]['content'] = (X.clone().detach(), None)

            # run detections with calibrated parameters
            param_dict = G.nodes[motif]['params']
            print(f'Extracting motif {motif} with calibrated params.')
            spikes, output = bfsw_detector_pt(scene, X, mask, **param_dict)

            #if not is_leaf:
            # set_trace()

            # Clean up
            G.nodes[motif]['extraction_dict'] = output
            # Put the incidence map on all outgoing edges (parents)
            for edge in G.in_edges(motif):
                parent = edge[0]
                G[parent][motif]['occurrence'] = torch.clone(spikes.detach())

            print(f'Motif {motif} extraction complete.')
            print(f'Spike map norm (want 1): {torch.sum(spikes)}')

    return torch.clone(spikes.detach())

def detect_hierarchical(Y, G, motif_list=None):
    """Detect a hierarchically-structured object in the scene Y.

    This function requires the graph G to contain calibrated or extracted
    motifs. See calibrate_extraction.py.

    Inputs:
    ---------
    Y - (M, N, C) numpy array
        Scene to look for hierarchically structured object in
    G - networkx graph object
        Hierarchy for the object to extract motifs from. 
        For our purposes, a hierarchy is a rooted directed acyclic graph (the
        root corresponds to the top-level "object"), with every node reachable
        from the root.
        Needs to have occurrence map motifs for composite motifs and parameter
        dicts set, e.g. by running calibrate_extraction prior.

    """

    from images import mask_to_bbox_pt
    from detection import (bfsw_detector_pt)

    # Get traversal order
    enumeration = enumerate_nodes_for_detection(G)
    dev = device()

    if motif_list is None:
        motif_list = [motif for motif in G.nodes]

    # Work loop: bottom to top.
    for depth_idx in range(len(enumeration)):
        for motif_idx in range(len(enumeration[depth_idx])):
            motif = enumeration[depth_idx][motif_idx]
            if motif_list.count(motif) > 0:
                print(f'Processing motif {motif}.')
            else:
                print(f'Motif {motif} not in motif list (debug):'
                        ' skipping.')
                continue

            # Check: is this motif a leaf?
            is_leaf = len(G[motif]) == 0

            # Set up the registration targets
            if is_leaf:
                # Setup the motif for leaf case
                X, mask = G.nodes[motif]['content']
                scene = Y
            else:
                # Setup the motif for composite case
                # Combine inbound occurrence maps
                num_chan = len(G[motif])
                occ_map = torch.zeros((num_chan, Y.shape[1], Y.shape[2]),
                        device=dev, dtype=precision())
                idx = 0
                for edge in G[motif]:
                    occ_map[idx, ...] = torch.clone(
                            G[motif][edge]['occurrence'][0, ...])
                    idx += 1
                    
                # Get target motif occurrence map
                X = G.nodes[motif]['occ_map']
                scene = occ_map
                mask = None

            # Get params
            param_dict = G.nodes[motif]['params'].copy()

            # Perform strided detection
            spikes, output = bfsw_detector_pt(scene, X, mask, **param_dict)

            # if not is_leaf:
            #     set_trace()


            # Clean up
            G.nodes[motif]['detection_dict'] = output
            # Put the incidence map on all outgoing edges (parents)
            for edge in G.in_edges(motif):
                parent = edge[0]
                G[parent][motif]['occurrence'] = torch.clone(spikes.detach())

            print(f'Motif {motif} detection complete.')
            print(f'Spike map norm (want 1): {torch.sum(spikes)}')
            # if not is_leaf:
            #     print('Sorted final errors:')
            #     print(np.sort(output['errors'][:,-1].to('cpu').numpy()))

    return spikes

def check_hierarchy(G):
    """Check the graph G to ensure it's a valid hierarchy

    Our hierarchy constraints are:
    1. G is a directed acyclic graph
    2. G is rooted (with G.graph['root'] denoting the root)
    3. Every node in G is reachable from the root node (in particular, G is
        connected)
    

    Inputs:
    ----------
    G - networkx DiGraph object
        The graph corresponding to the hierarchy to check

    Outputs:
    ----------
    is_valid_hierarchy - bool
        True or False depending on whether the graph is valid or not

    """

    import networkx as nx
    from networkx.algorithms.cycles import find_cycle
    from networkx.algorithms.components import is_weakly_connected

    # Check digraph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('Graph is not a directed graph.')
        return False

    # Check acyclic
    try:
        a = find_cycle(G)
        print('Graph is not acyclic.')
        print(f'(found cycle {a})')
        return False
    except nx.NetworkXNoCycle:
        # This means acyclic.
        pass

    # Check root
    try:
        root = G.graph['root']
    except KeyError:
        print('Graph is not rooted (define G.graph["root"] as the root node)')
        return False

    # Check connectivity
    return is_weakly_connected(G)

def enumerate_nodes_for_detection(G):
    """Get an enumeration of input hierarchy's nodes for extraction/detection

    Our extraction/detection algorithms work by traversing the hierarchy G in
    the following way:
    1. Start at low-level motifs (leaves), extract occurrence maps.
    2. Moving up the hierarchy, piece together any occurrence maps for
       low-level motifs into composite spike maps, and detect composite motifs
       (e.g. the crab's eye pair)
    3. Continue doing this until we reach the root / top level object.

    Traversing in this way efficiently requires a specific ordering of the
    nodes. In particular:
    1. We want to start at leaves, then do relevant composite motifs, all the
       way up until we reach the object. But we need to do the composite motifs
       in the right order (resolve all "dependencies" before we need to detect
       a higher-level composite object)
    2. The right way to do this in a rooted tree is to use breadth-first search
       to get the depth (distance from root) of each node in the hierarchy,
       then run detections in order of decreasing depth.
    3. The previous fails for general DAGs, though (there is e.g. a chance a
       low-level motif is included in a composite motif, and also included in
       the object by itself). In this setting, the right way to traverse is to
       compute the depth not by BFS, but by measuring the longest path from
       root to each node. This can be done efficiently in DAGs.

    This algorithm generates the enumeration specified in (3.) above.
    Its output is a list of lists: the outer list has length equal to the
    maximum depth in G; its first element is a list of the names of each node
    at maximum distance len( <the list> ) from the root, and so on.

    Inputs:
    ----------
    G - networkx DiGraph object
        The graph corresponding to the hierarchy to traverse

    Outputs:
    ----------
    enumeration - list of lists
        The enumeration, as specified in the last graf above.

    """

    import networkx as nx
    from networkx.algorithms.shortest_paths.weighted import (
            single_source_bellman_ford_path_length)

    is_valid = check_hierarchy(G)
    if not is_valid:
        raise ValueError('The input hierarchy G is not valid.' 
                ' (see print messages above for specific issue)')

    # Create an auxiliary graph to use for longest path traversal.
    G_aux = nx.DiGraph()
    for node in G.nodes:
        G_aux.add_node(node)
        for edge in G[node]:
            G_aux.add_edge(node, edge)
            G_aux[node][edge]['weight'] = -1

    # Shortest path in a DAG with edge weights -1 corresponds to longest path
    lengths = single_source_bellman_ford_path_length(G_aux, G.graph['root'])
    depth = 0
    for node in lengths.keys():
        lengths[node] *= -1
        if lengths[node] > depth:
            depth = lengths[node]

    # Prepare output
    enumeration = [[] for d in range(depth+1)]
    for node in lengths.keys():
        enumeration[depth - lengths[node]].append(node)

    return enumeration

def view_graph(G, key='content'):
    """Visualize a hierarchical object detection graph data structure's content

    See e.g. crab.py for basic structure info    

    
    Inputs:
    -----------
    G - networkx graph object
        Graph to be plotted. See e.g. crab.py's hierarchy function for an
        example
    key - (optional) string
        Gives the key to use for generating the image to plot at each node.
        Default is 'content', which is the key to get the motif/mask for the
        basic object graph. But a different key can be passed to plot e.g.
        intermediate detection output spike maps (etc)

    Outputs:
    ---------
    none. Plots the figure with plt.show() at the end of the function.

    NOTE: Requires pygraphviz.
        ` conda install -c conda-forge pygraphviz `

    NOTE: Used
    https://networkx.org/documentation/latest/auto_examples/drawing/plot_custom_node_icons.html#sphx-glr-auto-examples-drawing-plot-custom-node-icons-py
    as a source

    """

    from networkx.drawing.nx_agraph import graphviz_layout

    # Get a reproducible layout and create figure
    pos = nx.spring_layout(G, seed=1734289230)
    pos = graphviz_layout(G, prog='dot')
    fig, ax = plt.subplots()

    # Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
    # Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
    # but suppress arrowheads with `arrowstyle="-"`
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        min_source_margin=40,
        min_target_margin=40,
    )

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.00025
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        img = G.nodes[n]['content'][0]
        msk = G.nodes[n]['content'][1]
        png= np.concatenate((img,msk[...,None]), -1)
        a.imshow(png)
        a.axis("off")
        a.title.set_text(n)
    plt.show()

    pass

def embed_motif(Y, X, u, v):
    """Helper function: embed a motif in the provided scene at specific coords


    Inputs:
    ---------
    Y : (M, N, C) numpy array
        Scene to embed into. Just used to calculate pad sizes.
    X : (m, n, C) numpy array
        Motif to embed. Should be smaller than the scene.
    u : (1,) numpy array
        Vertical coordinate (zero at the top) for top-left motif pixel to embed
        at
    v : (1,) numpy array
        Horizontal coordinate (zero at the left) for top-left motif pixel to
        embed at

    Outputs:
    ---------
    (M, N, C) numpy array, with the embedded motif. Embeds on a black
    background.

    """

    u = int(u)
    v = int(v)

    M, N, C = Y.shape
    m, n, C = X.shape

    pad_top = u
    pad_bot = M - (pad_top + m)
    pad_left = v
    pad_right = N - (pad_left + n)

    embed = np.pad(X, ((pad_top, pad_bot), (pad_left, pad_right), (0,0)))

    return embed
