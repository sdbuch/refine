# Resource-Efficient Invariant Networks (REfINe)
## Neural network architectures for invariant computing with visual data, from unrolled optimization

This repository contains code for the arXiv paper "Resource-Efficient Invariant
Networks: Exponential Gains by Unrolled Optimization", by Sam Buchanan, Jingkai
Yan, Ellie Haber, and John Wright. If you find any code or implementations
useful, please [cite the paper](#Citation-information).

### Organization

The repository is organized as follows:
- `experiments`: Contains Jupyter notebooks corresponding to the four main
  figures in the paper (see [below](#Citation-information)). 
- `src`: Contains implementations for solvers and networks. For a
  solver-centric description of relevant pieces of the code, see
  [below](#Formulation-descriptions).
- `data`: Contains templates and backgrounds used in our experiments.

### Requirements

An `environment.yml` is provided -- PyTorch code has been tested on PyTorch
1.8.1 and CUDA 10.1, and the environment file reflects this. For other setup
methods, the only packages necessary are:
- NetworkX
- Numpy
- Scipy
- PyTorch
- Matplotlib

### Formulation descriptions

Below, we reference formula numbers in v1 of the arXiv preprint of the paper.
- The cost-smoothed formulation (3) with the euclidean transformation motion
  model is implemented in PyTorch at `registration_pt.reg_l2_rigid`, called
  with argument `image_type='textured'`.
- The with-background formulation (4) is implemented in Numpy at
  `registration_np.registration_l2_exp` for all four parametric motion models. 
- The complementary smoothing spike registration formulation (10) with the
  affine motion model is implemented in PyTorch at
  `registration_pt.reg_l2_spike`. A simplified version with the euclidean
  transformation motion model is implemented in PyTorch at
  `registration_pt.reg_l2_rigid`, called with argument `image_type='spike'`.


### Citation information
The arXiv link is [here](https://arxiv.org/abs/2203.05006). The bibtex information is:
```
@ARTICLE{buchanan-refine,
  title         = "{Resource-Efficient} Invariant Networks: Exponential Gains
                   by Unrolled Optimization",
  author        = "Buchanan, Sam and Yan, Jingkai and Haber, Ellie and Wright,
                   John",
  month         =  mar,
  year          =  2022,
  url           = "http://arxiv.org/abs/2203.05006",
  archivePrefix = "arXiv",
  eprint        = "2203.05006",
  primaryClass  = "cs.CV",
  arxivid       = "2203.05006"
}
```
