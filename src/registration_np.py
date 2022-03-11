#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import numpy as np
import numpy.linalg as npla
import scipy as sp
import matplotlib.pyplot as plt


def identity_vf(M, N, RM=None, RN=None):
    """Get vector field for the identity transformation.

    This returns the vector field (tau_u, tau_v) corresponding to the identity
    transformation, which maps the image plane to itself. 

    For more details on these vector fields, see the doc for affine_to_vf

    inputs:
    --------
    M : int
        vertical (number of rows) size of image plane being worked with
    N : int
        horizontal (number of cols) size of image plane being worked with
    RM : int (optional)
        number of points in the M direction desired. by default, this is M,
        giving the identity transformation. when a number other than M is
        provided, this corresponds to a resampling in the vertical direction.
        (we put this operation in this function because it is so naturally
        related)
    RN : int (optional)
        number of points in the N direction desired. by default, this is N,
        giving the identity transformation. when a number other than N is
        provided, this corresponds to a resampling in the horizontal direction.
        (we put this operation in this function because it is so naturally
        related)

    outputs:
    -------
    eu : numpy.ndarray (size (M, N))
        horizontal component of vector field corresponding to (I, 0)
    ev : numpy.ndarray (size (M, N))
        vertical component of vector field corresponding to (I, 0)
    
    """

    if RM is None:
        RM = M
    if RN is None:
        RN = N

    m_vec = np.linspace(0, M-1, RM)
    n_vec = np.linspace(0, N-1, RN)

    eu = np.dot(m_vec[:,np.newaxis], np.ones(RN)[:,np.newaxis].T)
    ev = np.dot(np.ones(RM)[:,np.newaxis], n_vec[:,np.newaxis].T)

    return (eu, ev)

def get_default_pgd_dict(**kwargs):
    """Get default parameter dictionary for proximal gradient descent solvers

    Valid key-value pairs are:

    init_pt = function with two arguments (m, n), two return values (numpy
              arrays of size (m, n))
        initial iterate to start GD at, represented as a function: must be a
        function with two arguments (m, n), the first of which represents image
        height and the second of which represents image width; and must return
        a tuple of two numpy arrays, each of size m, n, corresponding to the
        initial deformation field
    center : numpy array of shape (2,)
        Denotes an optional (set to np.array([0,0]) by default) center
        coordinate to use when solving the parametric version of the problem
        (parametric = True below). All affine transformations computed then
        have the form A * ( [i,j] - center ) + center + b, where A may have
        more structure if certain values of motion_model is
        set. This kind of reparameterization does not make a difference in the
        nonparametric version of the problem, so nothing is implemented for
        this case.
    sigma : float (positive)
        Bandwidth parameter in the gaussian filter used for the cost smoothing.
        (larger -> smaller cutoff frequency, i.e. more aggressive filtering)
        See gaussian_filter_2d
    sigma0 : float (positive)
        Bandwidth parameter in the gaussian filter used for complementary
        smoothing in registration_l2_spike.
        (larger -> smaller cutoff frequency, i.e. more aggressive filtering)
        See gaussian_filter_2d
    sigma_scene : float (positive)
        Bandwidth parameter in the gaussian filter used in scene smoothing in
        registration_l2_bbg.  (larger -> smaller cutoff frequency, i.e. more
        aggressive filtering) See gaussian_filter_2d
    window : NoneType or numpy array of size (m, n)
        Either None, if no window is to be used, or an array of size (m, n)
        (same as image size), denoting the cost window function to be applied
        (l2 error on residual is filtered, then windowed, before computing).
        NOTE: current implementation makes window independent of any setting of
        the parameter center specified above
    max_iter : int
        Maximum number of iterations to run PGD for
    tol : float (positive)
        Minimum relative tolerance before exiting optimization: optimization
        stops if the absolute difference between the loss at successive
        iterations is below this threshold.
    step : float (positive)
        Step size. Currently using constant-step gradient descent
    lam : float (positive)
        Regularization weight (multiplicative constant on the regularization
        term in the loss)
    use_nesterov : bool
        Whether or not to use Nesterov accelerated gradient descent
    use_restarting : bool
        Whether or not to use adaptive restarted Nesterov accelerated gradient
        descent. Speeds things up significantly, but maybe does not work well
        out of the box with proximal iteration
    motion_model : string (default 'nonparametric')
        Sets the motion model that the registration algorithm will use (i.e.
        what constraints are enforced on the transformation vector field).
        Values that are implemented are:
        'translation'
            transformation vector field is constrained to be translational (a
            pixel shift of the input). 2-dimensional.
        'rigid'
            transformation vector field is constrained to be a rigid motion / a
            euclidean transformation (i.e. a combination of a
            positively-oriented rotation and a translation). 3-dimensional.
        'similarity'
            transformation vector field is constrained to be a similarity
            transformation (i.e. a combination of a global dilation and a
            translation). 4-dimensional.
        'affine'
            transformation vector field is constrained to be an affine
            translation (i.e. a combination of a linear map and a translation).
            6-dimensional.
        'nonparametric'
            transformation vector field is allowed to be completely general,
            but regularization is added to the gradient descent solver via a
            complexity penalty, and the solver runs proximal gradient descent
            instead. (see e.g. entry for lambda for more info on associated
            parameters).
        TODO: Implement homography.
    gamma : float (min 0, max 1)
        Nesterov accelerated GD momentum parameter. 0 corresponds to the
        "usual" Nesterov AGD. 1 corresponds to "vanilla" GD. The optimal value
        for a given problem is the reciprocal condition number. Setting this to
        1 is implemented differently from setting use_nesterov to False (the
        algorithm is the same; but the former is slower)
    theta : float
        initial momentum term weight; typically 1
    precondition : bool
        Whether or not to use a preconditioner (divide by some scalars on each
        component of the gradient) for the A and b gradients in parametric
        motion models (see motion_model).. 
    epoch_len : int (positive)
        Length of an epoch; used for printing status messages
    quiet : bool
        If True, nothing will be printed while optimizing.
    record_movie : bool
        If True, a "movie" gets created from the optimization trajectory and
        logged to disk (see movie_fn param). Requires moviepy to be installed
        (easy with conda-forge). Potentially requires a ton of memory to store
        all the frames (all iterates)
    movie_fn : string
        If record_movie is True, this gives the location on disk where the
        movie will be saved
    movie_fps : int
        If record_movie is True, this gives the fps of the output movie.
    window_pad_size : int
        If record_movie is true, denotes the thickness of the border
        designating the window to be output in the movie
    frame_printing_stride : int
        If record_movie is true, denotes the interval at which log information
        will be written to the movie (every frame_printing_stride frames, log
        info is written; the actual movie fps is set by movie_fps above)
    font_size : int
        If record_movie is true, denotes the font size used for printing
        logging information to the output window. Set smaller for smaller-size
        images.

    NOTE: No value checking is implemented right now.
    
    Inputs:
    --------
    kwargs :
        any provided key-value pairs will be added to the parameter dictionary,
        replacing any defaults they overlap with

    Outputs:
    --------
    param_dict : dict
        dict of parameters to be used for a proximal gd solver. Pass these to
        e.g. nonparametric_registration or similar solvers.

    """

    param_dict = {}

    # Problem parameters: filter bandwidths, etc
    param_dict['sigma'] = 3
    param_dict['sigma_scene'] = 1.5
    param_dict['sigma0'] = 1
    param_dict['init_pt'] = lambda m, n: identity_vf(m, n)
    param_dict['motion_model'] = 'nonparametric'
    param_dict['window'] = None
    param_dict['center'] = np.zeros((2,))

    # Solver parameters: tolerances, stopping conditions, step size, etc
    param_dict['max_iter'] = int(1e4)
    param_dict['tol'] = 1e-4
    param_dict['step'] = 1
    param_dict['lam'] = 1
    param_dict['use_nesterov'] = False
    param_dict['use_restarting'] = False
    param_dict['gamma'] = 0
    param_dict['theta'] = 1
    param_dict['precondition'] = True

    # Logging parameters
    param_dict['epoch_len'] = 50
    param_dict['quiet'] = False
    param_dict['record_movie'] = False
    param_dict['movie_fn'] = ''
    param_dict['movie_fps'] = 30
    param_dict['window_pad_size'] = 5
    param_dict['frame_printing_stride'] = 10 # 3 times per second
    param_dict['font_size'] = 30
    param_dict['movie_gt'] = None
    param_dict['movie_proc_func'] = None

    # Legacy/compatibility stuff
    param_dict['parametric'] = False
    param_dict['translation_mode'] = False
    param_dict['rigid_motion_mode'] = False
    param_dict['similarity_transform_mode'] = False

    # Add user-provided params
    for arg in kwargs.keys():
        param_dict[arg] = kwargs[arg]

    return param_dict

def affine_to_vf(A, b, M, N):
    """Given (A, b), return associated vector field on M x N image plane

    An affine transformation is parameterized by an invertible matrix A and a
    vector b, and sends a 2D vector x to the 2D vector A*x + b. In the image
    context, x lies in the M by N image plane. This function takes the pair (A,
    b), and returns the associated vector field (tau_u, tau_v): here tau_u and
    tau_v are M by N matrices such that (tau_u)_{ij} = (1st row of A) * [i, j]
    + b_1, and (tau_v)_{ij} = (2nd row of A) * [i, j] + b_2. The matrices thus
    represent how the affine transformation (A, b) deforms the sampled image
    plane.

    Thus in general tau_u and tau_v have entries that may not be contained in
    the M by N image plane and may not be integers. These issues of boundary
    effects and interpolation effects are to be handled by other functions


    inputs:
    --------
    A : numpy.ndarray (size (2, 2))
        GL(2) part of affine transformation to apply
    b : numpy.ndarray (size (2,))
        translation part of affine transformation to apply
    M : int
        vertical (number of rows) size of image plane being worked with
    N : int
        horizontal (number of cols) size of image plane being worked with

    outputs:
    -------
    tau_u : numpy.ndarray (size (M, N))
        horizontal component of vector field corresponding to (A, b)
    tau_v : numpy.ndarray (size (M, N))
        vertical component of vector field corresponding to (A, b)

    """

    # Do it with broadcasting tricks (dunno if it's faster)
    A0 = A[:,0]
    A1 = A[:,1]
    eu = np.dot(np.arange(M)[:,np.newaxis], np.ones(N)[:,np.newaxis].T)
    ev = np.dot(np.ones(M)[:,np.newaxis], np.arange(N)[:,np.newaxis].T)

    tau = A0[np.newaxis, np.newaxis, :] * eu[..., np.newaxis] + \
            A1[np.newaxis, np.newaxis, :] * ev[..., np.newaxis] + \
            b[np.newaxis, np.newaxis, :] * np.ones((M, N, 1))

    # # For loop way
    # tau = np.zeros((M, N, 2))
    # for i in np.arange(M):
    #     for j in np.arange(N):
    #         tau[i,j,0] = A[0, 0] * i + A[0, 1] * j + b[0]
    #         tau[i,j,1] = A[1, 0] * i + A[1, 1] * j + b[1]

    return (tau[:,:,0], tau[:,:,1])

def vf_to_affine(tau_u, tau_v, ctr):
    """Get affine transformation corresponding to a vector field.

    General vector fields need not correspond to a particular affine
    transformation. In our formulation, we parameterize affine transforms as
    tau_u = a * (m-ctr[0] * \One)\One\\adj 
            + b * \One (n - ctr[1]*\One)\\adj 
            + (c + ctr[0]) * \One\One\\adj,
    and similarly for tau_v.

    We use the fact that this parameterization is used here to recover the
    parameters of the affine transform using simple summing/differencing.

    We need ctr as an input because the translation parameter is ambiguous
    without knowing the center. However, we can always recover the parameters
    of the transformation with respect to any fixed center (say, ctr = zero). 
    In general, if one provides ctr=np.zeros((2,)) to this function, it is a
    left inverse of affine_to_vf called with the correct M, N parameters.


    inputs:
    --------
    tau_u, tau_v : M by N numpy arrays
        u and v (resp.) components of the transformation field.
    ctr : (2,) shape numpy array
        center parameter that the transform was computed with. see center
        option in registration_l2. translation parameter is ambiguous without
        knowing the center.

    outputs:
    --------
    A : (2,2) numpy array
        The A matrix corresponding to the affine transform. Follows our
        conventions for how we compute with vector fields in determining how
        the entries of A are determined
    b : (2,) shape numpy array
        The translation parameter corresponding to the affine transform.
        Follows standard coordinates on the image plane (as elsewhere).

    """

    M, N = tau_u.shape

    a00 = tau_u[1, 0] - tau_u[0, 0]
    a01 = tau_u[0, 1] - tau_u[0, 0]
    a10 = tau_v[1, 0] - tau_v[0, 0]
    a11 = tau_v[0, 1] - tau_v[0, 0]

    A = np.array([[a00, a01], [a10, a11]])

    u_sum = np.sum(tau_u)
    v_sum = np.sum(tau_v)
    m_sum = np.sum(np.arange(M) - ctr[0] * np.ones((M,)))
    n_sum = np.sum(np.arange(N) - ctr[1] * np.ones((N,)))
    b0 = (u_sum - a00 * m_sum * N - a01 * M * n_sum) / M / N - ctr[0]
    b1 = (v_sum - a10 * m_sum * N - a11 * M * n_sum) / M / N - ctr[1]

    b = np.array([b0, b1])

    return A, b

def registration_l2_exp(Y, X, W, Om, center, transform_mode, optim_vars, param_dict=get_default_pgd_dict(), visualize=False):

    """ 

    This is yet another version of the cost-smoothed motif detection, in which we also infer 
     a (constant) background around the motif 


    Inputs: 
        Y -- input image
        X -- motif, embedded into an image of the same size as the target image
        Om -- support of the motif
        transform_mode -- 'affine', 'similarity', 'euclidean', 'translation'
    
    Outputs: 
        same as usual 

    """

    from time import perf_counter

    vecnorm_2 = lambda A: np.linalg.norm( A.ravel(), 2 )

    m, n, c = Y.shape    
    
    # Gradient descent parameters
    MAX_ITER = param_dict['max_iter']
    TOL = param_dict['tol']
    step = param_dict['step']

    if transform_mode == 'affine':
        [A, b] = optim_vars
    elif transform_mode == 'similarity':
        [dil, phi, b] = optim_vars
        A = dil * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    elif transform_mode == 'euclidean':
        [phi, b] = optim_vars
        A = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    elif transform_mode == 'translation':
        [b] = optim_vars
        A = np.eye(2)
    else:
        raise ValueError('Wrong transform mode.')

    # initialization (here, affine motion mode) 
    corr = np.dot(np.eye(2) - A, center)    
    tau_u, tau_v = affine_to_vf(A, b + corr, m, n)
        
    # External smoothing: calculate gaussian weights
    g = gaussian_filter_2d(m,n,sigma_u=param_dict['sigma'])    
    g = g / np.sum(g)
    
    h = gaussian_filter_2d(m,n,sigma_u=5*param_dict['sigma'])
    h = h / np.sum(h)
     
    # Calculate initial error
    error = np.inf * np.ones( (MAX_ITER,) )
    Rvals = np.zeros( (MAX_ITER,) )
               
    # initial interpolated image and error 
    cur_Y = image_interpolation_bicubic(Y, tau_u, tau_v )
    
    # initialize the background
    beta0 = cconv_fourier(h[...,np.newaxis], cur_Y - X) 
    #beta0 = np.zeros((m,n,c))
    beta = cconv_fourier(h[...,np.newaxis], beta0) 
    
    cur_X = np.zeros((m,n,c))
    cur_X = (1-Om)*beta + Om*X 
    
    FWres = W * cconv_fourier(g[...,np.newaxis], cur_Y-cur_X) 
    
    grad_A = np.zeros( (2,2) )
    grad_b = np.zeros( (2,) )
    m_vec = np.arange(m) - center[0]
    n_vec = np.arange(n) - center[1]

    if param_dict['use_nesterov'] is False:
        # print('Optimizing with vanilla PGD.')
        # for idx in range(1, MAX_ITER):
        for idx in range(MAX_ITER):
            
         
            # Get the basic gradient ingredients
            Y_dot_u = dimage_interpolation_bicubic_dtau1(Y, tau_u, tau_v)
            Y_dot_v = dimage_interpolation_bicubic_dtau2(Y, tau_u, tau_v)

            # Get the "tau gradient" part.
            # All the translation-dependent parts of the cost can be handled
            # here, so that the parametric parts are just the same as always.
            dphi_dY = cconv_fourier(dsp_flip(g)[...,np.newaxis], FWres) 
                                                   
            tau_u_dot = np.sum(dphi_dY * Y_dot_u, -1)
            tau_v_dot = np.sum(dphi_dY * Y_dot_v, -1)

            # Get parametric part gradients
            # Convert to parametric gradients

            # Get row and col sums
            tau_u_dot_rowsum = np.sum(tau_u_dot, 1)
            tau_u_dot_colsum = np.sum(tau_u_dot, 0)
            tau_v_dot_rowsum = np.sum(tau_v_dot, 1)
            tau_v_dot_colsum = np.sum(tau_v_dot, 0)

            # Put derivs
            # These need to be correctly localized to the region of interest
            grad_A[0, 0] = np.dot(tau_u_dot_rowsum, m_vec)
            grad_A[1, 0] = np.dot(tau_v_dot_rowsum, m_vec)
            grad_A[0, 1] = np.dot(tau_u_dot_colsum, n_vec)
            grad_A[1, 1] = np.dot(tau_v_dot_colsum, n_vec)

            grad_b[0] = np.sum(tau_u_dot_rowsum)
            grad_b[1] = np.sum(tau_v_dot_rowsum)

            # Precondition
            grad_A /= 100

            
            dphi_dbeta0 = -cconv_fourier( dsp_flip(h)[...,np.newaxis], (1-Om) * dphi_dY )

            # Now update parameters
            grad_norm = np.sqrt(npla.norm(grad_A.ravel(),2)**2 + npla.norm(grad_b,ord=2)**2)

            #phi = phi - step * grad_phi / 86
            if idx > 5:
                if transform_mode == 'affine':
                    A     = A - step * grad_A
                    b     = b - step * grad_b

                elif transform_mode == 'similarity':
                    grad_dil, grad_phi, grad_b = l2err_sim_grad(dil, phi, grad_A, grad_b)
                    dil   = dil - step * grad_dil * 0.1
                    phi   = phi - step * grad_phi
                    b     = b - step * grad_b

                    A = dil * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

                elif transform_mode == 'euclidean':
                    grad_phi, grad_b = l2err_se_grad(phi, grad_A, grad_b)
                    phi   = phi - step * grad_phi
                    b     = b - step * grad_b
                    A = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

                elif transform_mode == 'translation':
                    b     = b - step * grad_b
                    A = np.eye(2)
                

                # # Constraint on singular values
                # u,s,vh = npla.svd(A)
                # for i_sv in range(len(s)):
                #     if abs(s[i_sv]) < svlim[0]:
                #         s[i_sv] = np.sign(s[i_sv]) * svlim[0]
                #     if abs(s[i_sv]) > svlim[1]:
                #         s[i_sv] = np.sign(s[i_sv]) * svlim[1]
                # A = u * np.diag(s) * vh

            beta0 = beta0 - 25 * step * dphi_dbeta0

                
            corr = np.dot(np.eye(2) - A, center)
            tau_u, tau_v = affine_to_vf(A, b + corr, m, n)

            # Bookkeeping (losses and exit check)
            cur_Y = image_interpolation_bicubic(Y, tau_u, tau_v ) 

            beta = cconv_fourier(h[...,np.newaxis], beta0) 
            
            #print('beta!')
            #plt.imshow(beta)
            #plt.show()
            #plt.imshow(dphi_dbeta0)
            #plt.show()

            cur_X = np.zeros((m,n,c))
            cur_X = (1-Om)*beta + Om*X 

            FWres = W * cconv_fourier(g[...,np.newaxis], cur_Y-cur_X) 

            error[idx] = .5 * np.sum(FWres ** 2)

            # cur_X_wd = (cur_X - np.mean(cur_X * W, axis=(0,1))) * W
            # cur_Y_wd = (cur_Y - np.mean(cur_Y * W, axis=(0,1))) * W

            cur_X_wd = cur_X * Om
            for ic in range(3):
                cur_X_wd[:,:,ic] -= np.mean(cur_X_wd[:,:,ic][cur_X_wd[:,:,ic] > 0])
            cur_Y_wd = cur_Y * Om
            for ic in range(3):
                cur_Y_wd[:,:,ic] -= np.mean(cur_Y_wd[:,:,ic][cur_Y_wd[:,:,ic] > 0])

            Rvals[idx] = np.sum(Om * cur_X_wd * cur_Y_wd) / ( vecnorm_2(Om * cur_X_wd) * vecnorm_2(Om * cur_Y_wd) )
        
            
            if idx > 0 and error[idx] > error[idx-1]:
                # print('Nonmontone, cutting step')
                step = step / 2 
            else:
                step = step * 1.01
                                    
            cur_Y_disp = cur_Y.copy()
            cur_Y_disp[:,:,1] = Om[:,:,1]
            cur_Y_disp[:,:,2] = Om[:,:,2]
            
            loopStop = perf_counter()
            
            if grad_norm < TOL:
                if param_dict['quiet'] is False:
                    print(f'Met objective at iteration {idx}, '
                            'exiting...')
              
                break
                

            if (idx % param_dict['epoch_len']) == 0:
                if param_dict['quiet'] is False:
                    # print(f'iter {idx}  obj {error[idx]}  A {A} svs {np.linalg.svd(A)} b {b}  grad_A {grad_A}  grad_b {grad_b} grad_beta {npla.norm(dphi_dbeta0.ravel(),2)} ')
                    print('iter {:d}      objective {:.4e}      correlation {:.4f}'.format(idx, error[idx], Rvals[idx]))
            
            if visualize is True:
                if (idx % 10) == 0:
                    if param_dict['quiet'] is False:
                        # f, (ax1, ax2) = plt.subplots(1, 2)
                        
                        # ax1.imshow(cur_Y_disp) 
                        # ax2.imshow(cur_X)
                        # plt.show()

                        plt.imshow(cur_Y_disp)
                        plt.show()


                    #print(f'phi-grad {grad_phi} \t\t\t b-grad {grad_b}...')
                    #plt.imshow(cur_Y)
                    #plt.show()
                    #plt.imshow(X)
                    #plt.show()
                    
                    #print(beta)
                    
                    """
                    plt.imshow(Y)
                    
                    bb_u1 = tau_u[bb_t,bb_l]
                    bb_v1 = tau_v[bb_t,bb_l]
                    
                    bb_u2 = tau_u[bb_t+roi_u-1,bb_l]
                    bb_v2 = tau_v[bb_t+roi_u-1,bb_l]

                    bb_u3 = tau_u[bb_t+roi_u-1,bb_l+roi_v-1]
                    bb_v3 = tau_v[bb_t+roi_u-1,bb_l+roi_v-1]

                    bb_u4 = tau_u[bb_t,bb_l+roi_v-1]
                    bb_v4 = tau_v[bb_t,bb_l+roi_v-1]

                    plt.plot( [ bb_v1_0, bb_v2_0, bb_v3_0, bb_v4_0, bb_v1_0 ], [ bb_u1_0, bb_u2_0, bb_u3_0, bb_u4_0, bb_u1_0 ], color = 'b', linewidth = 1 )
                    plt.plot( [ bb_v1, bb_v2, bb_v3, bb_v4, bb_v1 ], [ bb_u1, bb_u2, bb_u3, bb_u4, bb_u1 ], color = 'r', linewidth = 3 )
                    plt.show()"""

    # This next block of code is for Nesterov accelerated GD.
    else:
        raise NotImplementedError('Test function only implements vanilla GD')
        
    # plt.imsave('registration.png',np.maximum(np.minimum(cur_Y_disp,0.999),.0001))
    # plt.imsave('X_plus_beta.png',np.maximum(np.minimum(cur_X,0.999),.0001))

    if transform_mode == 'affine':
        optim_vars_new = [A, b]
    elif transform_mode == 'similarity':
        optim_vars_new = [dil, phi, b]
    elif transform_mode == 'euclidean':
        optim_vars_new = [phi, b]
    elif transform_mode == 'translation':
        optim_vars_new = [b]

    return tau_u, tau_v, optim_vars_new, error, Rvals

def dilate_support(Om,sigma):
    M = Om.shape[0]
    N = Om.shape[1]
    
    psi = gaussian_filter_2d(M,N,sigma_u=sigma)
    delta = np.exp(-2) * ((2.0*np.pi*sigma) ** -.5)
    Om_tilde = cconv_fourier(psi[...,np.newaxis],Om)
    for i in range(M):
        for j in range(N):
            if Om_tilde[i,j,0] < delta:
                Om_tilde[i,j,0] = 0
                Om_tilde[i,j,1] = 0
                Om_tilde[i,j,2] = 0
            else:
                Om_tilde[i,j,0] = 1
                Om_tilde[i,j,1] = 1
                Om_tilde[i,j,2] = 1

    return Om_tilde

def rotation_mat(theta):
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    mat = np.array([[cos, -sin], [sin, cos]])
    return mat

def l2err_se_grad(phi, grad_A, grad_b):
    """ Calculate loss gradient in SE registration prob using aff gradient

    This gradient is for the parametric version of the problem, with the
    parameterization in terms of the special euclidean group (oriented rigid
    motions of the plane).
    It wraps l2err_aff_grad, since chain rule lets us easily calculate this
    problem's gradient using the affine problem's gradient.

    Implementation ideas:
    - for ease of implementation, require the current angle phi as an input,
      although it could probably be determined from tau_u and tau_v in general.

    Inputs:
        phi : angle parameter of matrix part of current rigid motion iterate.
        grad_A : gradient of the cost with respect to A (matrix parameter of
            affine transform) (output from l2err_aff_grad)
        grad_b : gradient of the cost with respect to b (translation parameter
            of affine transform) (output from l2err_aff_grad)
    
    Outputs:
        grad_phi : gradient of the cost with respect to phi (angular parameter of
            rotational part of special euclidean transform:
        grad_b : gradient of the cost with respect to b (translation parameter
            of rigid motion)

    """

    # rigid motion derivative matrix
    G = np.array([[-np.sin(phi), -np.cos(phi)], [np.cos(phi), -np.sin(phi)]])

    # Put derivatives
    grad_phi = np.sum(G * grad_A)

    return grad_phi, grad_b

def l2err_sim_grad(dil, phi, grad_A, grad_b):
    """ Calculate loss gradient in similarity xform registration prob

    This gradient is for the parametric version of the problem, with the
    parameterization in terms of the similarity transformations (rigid motions
    with the rotation multiplied by a scale parameter).

    It wraps l2err_aff_grad, since chain rule lets us easily calculate this
    problem's gradient using the affine problem's gradient.

    Implementation ideas:
    - for ease of implementation, require the current angle phi as an input,
      although it could probably be determined from tau_u and tau_v in general.

    Inputs:
        dil : dilation (scale) parameter of matrix part of current similarity
            transform iterate.
        phi : angle parameter of matrix part of current rigid motion iterate.
        grad_A : gradient of the cost with respect to A (matrix parameter of
            affine transform) (output from l2err_aff_grad)
        grad_b : gradient of the cost with respect to b (translation parameter
            of affine transform) (output from l2err_aff_grad)
    
    Outputs:
        grad_phi : gradient of the cost with respect to dil (dilation/scale
            parameter of similarity transform)
        grad_phi : gradient of the cost with respect to phi (angular parameter of
            rotational part of special euclidean transform:
        grad_b : gradient of the cost with respect to b (translation parameter
            of rigid motion)

    """

    # rigid motion matrix
    G = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # rigid motion derivative matrix
    Gdot = np.array([[-np.sin(phi), -np.cos(phi)], [np.cos(phi), -np.sin(phi)]])

    # Put derivatives
    grad_dil = np.sum(G * grad_A)
    grad_phi = dil * np.sum(Gdot * grad_A)

    return grad_dil, grad_phi, grad_b

def apply_random_transform( X0, Om0, c, mode, s_dist, phi_dist, theta_dist, b_dist, return_params=True ):

    N0 = X0.shape[0]
    N1 = X0.shape[1]
    C = X0.shape[2]
    
    tf_params = sample_random_transform( mode, s_dist, phi_dist, theta_dist, b_dist )
    A = tf_params[0]
    b = tf_params[1]

    # apply the transformation 
    corr = np.dot(np.eye(2) - A, c)
    (tau_u, tau_v) = affine_to_vf(A, b + corr, N0, N1)

    X = image_interpolation_bicubic(X0, tau_u, tau_v)
    Om = image_interpolation_bicubic(Om0, tau_u, tau_v)
        
    if return_params is False:
        return X, Om
    else:
        return X, Om, tf_params

def sample_random_transform( mode, s_dist, phi_dist, theta_dist, b_dist ):
    
    s_min = s_dist[0]
    s_max = s_dist[1]
    phi_min = phi_dist[0]
    phi_max = phi_dist[1]
    theta_min = theta_dist[0]
    theta_max = theta_dist[1]
    b_min = b_dist[0]
    b_max = b_dist[1]
    
    b = np.zeros((2,))
    b[0] = np.random.uniform(b_min,b_max)
    b[1] = np.random.uniform(b_min,b_max)
    
    if mode == 'affine':
        s1 = np.random.uniform(s_min,s_max)
        s2 = np.random.uniform(s_min,s_max)
        phi = np.random.uniform(phi_min,phi_max)
        theta = np.random.uniform(theta_min,theta_max)
        
        U = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S = np.diag([s1, s2])
        A = np.matmul( U, np.matmul(S,V.transpose() ) )
        return [A, b, None, None]
        
    elif mode == 'similarity':
        dil = np.random.uniform(s_min,s_max)
        phi = np.random.uniform(phi_min,phi_max)
        
        A = dil * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        return [A, b, dil, phi]
        
    elif mode == 'euclidean':
        phi = np.random.uniform(phi_min,phi_max)
        
        A = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        return [A, b, None, phi]
        
    elif mode == 'translation':
        A = np.eye(2)
        return [A, b, None, None]

def image_interpolation_bicubic(x,tau1,tau2):
    """Resample an input image x, using bicubic interpolation
    
       We write 
    
           x'_{ij} = \sum_{kl} x_{kl} \\phi( \\tau1_{kl} - k ) \\phi( \\tau2_{kl} - l )
    
       where 
    
           phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
                    { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
       
       is the cubic convolution interpolant
    
    Inputs: 
       x - N0 x N1 x N2 input image 
       tau1 - M0 x M1, first component of the transformation
       tau2 - M0 x M1, second component of the transformation
    
    Outputs:
       x_pr - M0 x M1 x N2, resampled image
    
    Note: where necessary, we assume that x is extended by zero, i.e., we treat
        x_{kl} = 0 whenever k < 0, k >= N0, l < 0, or l >= N1

    Note: this function has the property that for basic slice objects s0, s1,
        one has f(x, tau1, tau2)[s0, s1] = f(x, tau1[s0, s1], tau2[s0, s1]).
    
    """

    
    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau1 - t1_0
    a0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau1 - t1_1
    a1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau1 - t1_2
    a2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau1 - t1_3
    a3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau2 - t2_0
    b0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau2 - t2_1
    b1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau2 - t2_2
    b2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau2 - t2_3
    b3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    x_pr = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    return x_pr

def dimage_interpolation_bicubic_dtau1(x,tau1,tau2):
    """Differentiates the bicubic interpolation
           x'_{ij} = \sum_{kl} x_{kl} \phi( \tau1_{kl} - k ) \phi( \tau2_{kl} - l )
       where 
           phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
                    { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
       with respect to the first component \tau1. This corresponds to the formula
    
           dx'_dtau1 = \sum_{kl} x_{kl} \phi_dot( \tau1_{kl} - k ) \phi( \tau2_{kl} - l )
    
       where
           phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
                        { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)
    
    Inputs: 
       x - N0 x N1 x N2 input image 
       tau1 - M0 x M1, first component of the transformation
       tau2 - M0 x M1, second component of the transformation
    
    Outputs:
       dx_pr_dtau1 - M0 x M1 x N2, derivative of resampled image 
    
    Note: where necessary, we assume that x is extended by zero, i.e., we treat x_{kl} = 0 whenever 
           k < 0, k >= N0, l < 0, or l >= N1

    Note: this function has the property that for basic slice objects s0, s1,
        one has f(x, tau1, tau2)[s0, s1] = f(x, tau1[s0, s1], tau2[s0, s1]).
    
    """
    

    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #

    # phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
    #              { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (sgn(u) = 1)
    u = tau1 - t1_0
    a0 = -1.5 * u ** 2 + 5 * u - 4
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (sgn(u) = 1)
    u = tau1 - t1_1
    a1 = 4.5 * u ** 2 - 5 * u 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (sgn(u) = -1)
    u = tau1 - t1_2
    a2 = -4.5 * u ** 2 - 5 * u 
 
    # 3: here, we are in case (1)
    #          and u is negative (sgn(u) = -1)
    u = tau1 - t1_3
    a3 = 1.5 * u ** 2 + 5 * u + 4 
    
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau2 - t2_0
    b0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau2 - t2_1
    b1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau2 - t2_2
    b2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau2 - t2_3
    b3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    dx_pr_dtau1 = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    
    return dx_pr_dtau1
    
def dimage_interpolation_bicubic_dtau2(x,tau1,tau2):
    """Differentiates the bicubic interpolation
           x'_{ij} = \sum_{kl} x_{kl} \phi( \tau1_{kl} - k ) \phi( \tau2_{kl} - l )
       where 
           phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
                    { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
       with respect to the first component \tau2. This corresponds to the formula
    
           dx'_dtau2 = \sum_{kl} x_{kl} \phi( \tau1_{kl} - k ) \phi_dot( \tau2_{kl} - l )
    
       where
           phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
                        { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)
    
    Inputs: 
       x - N0 x N1 x N2 input image 
       tau1 - M0 x M1, first component of the transformation
       tau2 - M0 x M1, second component of the transformation
    
    Outputs:
       dx_pr_dtau2 - M0 x M1 x N2, derivative of resampled image 
    
    Note: where necessary, we assume that x is extended by zero, i.e., we treat x_{kl} = 0 whenever 
           k < 0, k >= N0, l < 0, or l >= N1

    Note: this function has the property that for basic slice objects s0, s1,
        one has f(x, tau1, tau2)[s0, s1] = f(x, tau1[s0, s1], tau2[s0, s1]).

    """
    
    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau1 - t1_0
    a0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau1 - t1_1
    a1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau1 - t1_2
    a2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau1 - t1_3
    a3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2

    # phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
    #              { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)    
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (sgn(u) = 1)
    u = tau2 - t2_0
    b0 = -1.5 * u ** 2 + 5 * u - 4 
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (sgn(u) = 1)
    u = tau2 - t2_1
    b1 = 4.5 * u ** 2 - 5 * u 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (sgn(u) = -1)
    u = tau2 - t2_2
    b2 = -4.5 * u ** 2 - 5 * u
 
    # 3: here, we are in case (1)
    #          and u is negative (sgn(u) = -1)
    u = tau2 - t2_3
    b3 = 1.5 * u ** 2 + 5 * u + 4
    
    dx_pr_dtau2 = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    
    return dx_pr_dtau2
    
def apply_affine_transform( X0, Om0, c, A, b ):
    
    N0 = X0.shape[0]
    N1 = X0.shape[1]
    C = X0.shape[2]
    
    corr = np.dot(np.eye(2) - A, c)
    (tau_u, tau_v) = affine_to_vf(A, b + corr, N0, N1)

    X = image_interpolation_bicubic(X0, tau_u, tau_v)
    Om = image_interpolation_bicubic(Om0, tau_u, tau_v)
    
    return X, Om

def cconv_fourier(x, y):
    """Compute the circulant convolution of two images in Fourier space.

    Implementing this on its own because scipy.signal.fftconvolve seems to
    handle restriction in its 'same' mode incorrectly
    
    This function is implemented to work with potentially many-channel images:
    it will just perform the 2D convolution on the *first two dimensions* of
    the inputs. So permute dims if data is such that batch size/etc is first...

    Requires:
    x and y need to have the same shape / be broadcastable. (no automatic
    padding)

    """


    F_X = np.fft.fft2(x, axes=(0, 1), norm='backward')
    F_Y = np.fft.fft2(y, axes=(0, 1), norm='backward')

    F_XY = F_X * F_Y

    return np.real(np.fft.ifft2(F_XY, axes=(0, 1)))

def gaussian_filter_1d(N, sigma=1, offset=0):
    """Return a 1D gaussian filter with length N and inverse bandwidth sigma

    The filter is normalized to have unity value at DC (following an
    unnormalized fourier transform).

    Use circulant boundary conditions, with a phase shift to center the filter
    at index 0. The filter is generated to have "dsp flip" symmetry (see
    dsp_flip) regardless of parity of N.

    offset denotes an optional offset parameter, designating where the filter
    is centered -- gets taken modulo N

    """

    i = np.arange(0, N)

    g = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-((i - offset + (N-1)/2) % N -
        (N-1)/2)**2 / 2/ sigma**2)

    return g / npla.norm(g,ord=1)

def gaussian_filter_2d(M, N=None, sigma_u=1, sigma_v=None, offset_u = 0,
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
    gi = gaussian_filter_1d(M, sigma=sigma_u, offset=offset_u)
    gj = gaussian_filter_1d(N, sigma=sigma_v, offset=offset_v)

    # Expand dimensions for outer product...

    return gi[:,np.newaxis].dot(gj[:,np.newaxis].T)

def dsp_flip(X):
    """Compute the 'dsp flip' of input numpy tensor X

    If X[i1, ..., ik] represents the input tensor, this function returns the
    'dsp flipped' tensor X[-i1, ..., -ik], where all indexing is done modulo
    the sizes of each individual dimension of X. So compared to the usual
    flipud/fliplr (for matrices) flip, this leaves the first element in-place.

    Inputs:
    X : numpy array of any size

    Outputs:
    X with each dimension 'dsp flipped' as described above. Output type may be
    float (not same as X)

    """
    
    Ndims = len(X.shape)
    ax = tuple(range(Ndims)) * 2
    # what's a log factor between friends?
    return np.real(np.fft.fft2(X, axes=ax, norm='ortho'))

def generate_data(mode, aff_A=None, aff_b=None):
    s_dist = [0.8, 1.25]
    phi_dist = [-np.pi/4, np.pi/4]
    theta_dist = [-np.pi/4, np.pi/4]
    b_dist = [-5, 5]

    beach_bg = plt.imread('../data/beach_bg.jpg')
    beach_bg = 1/255 * beach_bg.astype('float64')

    a = np.random.randint(0,beach_bg.shape[0]-500)
    b = np.random.randint(0,beach_bg.shape[1]-600)
    beach_bg = beach_bg[a:a+500,b:b+600,:].copy()
    M, N, C = beach_bg.shape

    crabTight = plt.imread('../data/crab_big_bbg_tight.png')
    crabTight = crabTight[..., 0:3]
    Mc, Nc, Cc = crabTight.shape

    u = 160 
    v = 160 

    crab = np.zeros((M,N,C))
    crab[u:(u+Mc),v:(v+Nc),:] = crabTight
    crab_mask = (np.sum(crab, 2) > 0).astype('float64')[..., np.newaxis]


    c = np.zeros((2,))
    c[0] = 280
    c[1] = 280

    if aff_A is not None and aff_b is not None:
        crabDef, maskDef = apply_affine_transform(crab, crab_mask, c, aff_A, aff_b)
    else:
        crabDef,maskDef,tf_params = apply_random_transform(crab,crab_mask,c,
                                                    mode,s_dist,phi_dist,theta_dist,b_dist)
    #plt.imshow(crabDef)

    X = (1-maskDef) * beach_bg + maskDef * crabDef


    # generate our motif and its mask 
    body = plt.imread('../data/crab_body.png')
    body = body[:,:,0:3].copy()
    Mb, Nb, Cb = body.shape

    ub = 238
    vb = 192

    X0 = np.zeros((M,N,C))
    X0[ub:(ub+Mb),vb:(vb+Nb),:] = body
    Om = (np.sum(X0, 2) > 2e-1).astype('float64')[..., np.newaxis]

    Om_pr = np.zeros((M,N,C))
    for i in range(C):
        Om_pr[:,:,i] = Om[:,:,0].copy()
    Om = Om_pr.copy()

    psi = gaussian_filter_2d(M,N,sigma_u=1)
    X = cconv_fourier(psi[...,np.newaxis],X)
    
    if aff_A is not None and aff_b is not None:
        return X0, X, Om, c
    else:
        return X0, X, Om, c, tf_params, [s_dist,phi_dist,theta_dist,b_dist]

def test_complexity_textured():

    test_complexity_textured_run_data('translation')
    test_complexity_textured_run_data('euclidean')
    test_complexity_textured_run_data('similarity')
    test_complexity_textured_run_data('affine', sigma_init=10)

    target_corr = 0.9

    iter_recs = np.zeros((2,4))
    # time_recs = np.zeros((2,4))
    incomplete = np.zeros((2,4))

    for idx in range(1,11):
        
        with open('exp_affine/{:02d}_optim.pkl'.format(idx), "rb") as f:
            [Rvals_optim, _, elapsed_optim, _] = pickle.load(f)
        with open('exp_affine/{:02d}_cover.pkl'.format(idx), "rb") as f:
            [Rvals_cover, elapsed_cover] = pickle.load(f)
        
        good_id = np.where(Rvals_optim > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[0,0] += good_id[0] + 1
            # time_recs[0,0] += (good_id[0]+1) / len(Rvals_optim) * elapsed_optim
        else:
            incomplete[0,0] = 1
            
        good_id = np.where(Rvals_cover > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[1,0] += good_id[0] + 1
            # time_recs[1,0] += (good_id[0]+1) / len(Rvals_cover) * elapsed_cover
        else:
            incomplete[1,0] = 1
            
            
        with open('exp_similarity/{:02d}_optim.pkl'.format(idx), "rb") as f:
            [Rvals_optim, _, elapsed_optim, _] = pickle.load(f)
        with open('exp_similarity/{:02d}_cover.pkl'.format(idx), "rb") as f:
            [Rvals_cover, elapsed_cover] = pickle.load(f)
        
        good_id = np.where(Rvals_optim > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[0,1] += good_id[0] + 1
            # time_recs[0,1] += (good_id[0]+1) / len(Rvals_optim) * elapsed_optim
        else:
            incomplete[0,1] = 1
            
        good_id = np.where(Rvals_cover > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[1,1] += good_id[0] + 1
            # time_recs[1,1] += (good_id[0]+1) / len(Rvals_cover) * elapsed_cover
        else:
            incomplete[1,1] = 1
            
            
        with open('exp_euclidean/{:02d}_optim.pkl'.format(idx), "rb") as f:
            [Rvals_optim, _, elapsed_optim, _] = pickle.load(f)
        with open('exp_euclidean/{:02d}_cover.pkl'.format(idx), "rb") as f:
            [Rvals_cover, elapsed_cover] = pickle.load(f)
        
        good_id = np.where(Rvals_optim > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[0,2] += good_id[0] + 1
            # time_recs[0,2] += (good_id[0]+1) / len(Rvals_optim) * elapsed_optim
        else:
            incomplete[0,2] = 1
            
        good_id = np.where(Rvals_cover > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[1,2] += good_id[0] + 1
            # time_recs[1,2] += (good_id[0]+1) / len(Rvals_cover) * elapsed_cover
        else:
            incomplete[1,2] = 1
            
            
        with open('exp_translation/{:02d}_optim.pkl'.format(idx), "rb") as f:
            [Rvals_optim, _, elapsed_optim, _] = pickle.load(f)
        with open('exp_translation/{:02d}_cover.pkl'.format(idx), "rb") as f:
            [Rvals_cover, elapsed_cover] = pickle.load(f)
        
        good_id = np.where(Rvals_optim > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[0,3] += good_id[0] + 1
            # time_recs[0,3] += (good_id[0]+1) / len(Rvals_optim) * elapsed_optim
        else:
            incomplete[0,3] = 1
            
        good_id = np.where(Rvals_cover > target_corr)[0]
        if len(good_id) > 0:
            iter_recs[1,3] += good_id[0] + 1
            # time_recs[1,3] += (good_id[0]+1) / len(Rvals_cover) * elapsed_cover
        else:
            incomplete[1,3] = 1
            
            
            
    iter_recs /= 10
    time_recs /= 10
    iter_recs = iter_recs[:, ::-1]
    time_recs = time_recs[:, ::-1]

    iter_recs[0,:] = iter_recs[0,:] * np.array([[4,4,5,8]])
            
    print('Incomplete runs:')
    print(incomplete)
    print('Average complexity:')
    print(iter_recs)
