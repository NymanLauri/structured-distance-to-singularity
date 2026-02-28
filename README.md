# Structured distance to singularity
This repository contains code for finding the linearly structured distance to singularity.

The code requires that [Manopt](https://www.manopt.org/downloads.html) is downloaded and set in your path. Manopt is used for the Augmented Lagrangian option as well as certain tensor operations in the code.

This code has been tested with Manopt 8.0 and Matlab 2025b.

The file ``svdmin.m``, written by Ethan N. Epperly, Yuji Nakatsukasa and Taejun Park, is based on the work "Fast, High-Accuracy, Randomized Nullspace Computations for Tall Matrices", 2026, https://arxiv.org/abs/2602.16797 

The file ``example_toeplitz.m`` shows an example of how to run the algorithm.

Calling ``[Delta,iteration_count] = structured_distance_to_singularity(A,P)`` finds a perturbation Delta that respects the linear structure given in P such that A+Delta is singular. Here, P is a tensor such that P(:,:,i) contains the ith orthonormal basis matrix of the linear structure.

Calling ``[Delta,iteration_count] = sparse_distance_to_singularity(A)`` finds a perturbation Delta that respects the sparsity pattern of A such that A+Delta is singular. This implementation has been optimized for sparse structures. 
