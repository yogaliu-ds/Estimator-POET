# POET Model for Sparse Covariance Matrix Estimation

## Objective
Follow the paper: Large covariance estimation by thresholding principal orthogonal complements
Create unbiased precision matrix based on small time length.

This repository includes a Python module implementing the POET (PCA-based Optimal Estimation of Thresholds) model for estimating sparse covariance matrices. The implementation focuses on optimizing parameters and reconstructing data to estimate precision matrices effectively.
Overview

The module includes functions to perform the following tasks:

    Estimate the Optimal Number of Principal Components (estimator_POET)
    Split Data into Training and Validation Sets (data_split)
    Find the Minimum Threshold (find_c_min)
    Find the Optimal Threshold (find_c_star)
    Compute the Precision Matrix (compute_precision_matrix)

Functions
estimator_POET(p, T, demeaned_matrix, M, ic)

Objective: Determine the optimal number of principal components KK for the given data.

Parameters:

    p: Number of variables (nodes).
    T: Number of time steps.
    demeaned_matrix: The matrix of demeaned return series.
    M: Maximum number of principal components to consider.
    ic: A function for information criterion.

Returns:

    min_K: The optimal number of principal components.

data_split(Y, k)

Objective: Split the demeaned matrix into training and validation sets.

Parameters:

    Y: Demeaned data matrix.
    k: Number of principal components used for PCA.

Returns:

    train_u: Training set with reconstruction errors.
    val_u: Validation set with reconstruction errors.

find_c_min(train_u, M=1, step=0.005)

Objective: Find the minimum threshold CC that ensures the positive definiteness of the covariance matrix.

Parameters:

    train_u: Training set with reconstruction errors.
    M: Maximum threshold to consider.
    step: Increment for the threshold values.

Returns:

    C_min: The minimum threshold value.

find_c_star(train_u, val_u, start_point=0.055, end_point=0.065, step=0.001)

Objective: Find the optimal threshold CC that minimizes the Frobenius norm between the estimated and actual covariance matrices.

Parameters:

    train_u: Training set with reconstruction errors.
    val_u: Validation set with reconstruction errors.
    start_point: Starting point for the threshold values.
    end_point: Ending point for the threshold values.
    step: Increment for the threshold values.

Returns:

    C_star: The optimal threshold value.
    min_value: The minimum value of the Frobenius norm.

compute_precision_matrix(demeaned_matrix, C_star, round_decimal=7, num_eigenvalues=10)

Objective: Compute the precision matrix from the demeaned data matrix using the optimal threshold C∗C∗.

Parameters:

    demeaned_matrix: The matrix of demeaned return series.
    C_star: The optimal threshold value.
    round_decimal: Decimal precision for rounding the covariance matrix.
    num_eigenvalues: Number of eigenvalues to consider.

Returns:

    precision_matrix: The computed precision matrix.
