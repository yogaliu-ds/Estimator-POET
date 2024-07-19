import numpy as np
import pandas as pd
from preprocessing import process_data
from utils import ic1, ic2
from model import estimator_POET, data_split, find_c_min, find_c_star, compute_precision_matrix

# Load Data
'''
Input: Price series, shape=(T, N) 
'''
sdf = pd.read_csv('data/SP500_Price.csv', index_col=0)  # Example input

# Data Processing
demeaned_matrix, cov_matrix, T0 = process_data(sdf, training_ratio=0.8)
Y = demeaned_matrix

'''
IC: ic1, ic2
M: Time length
'''
min_k = estimator_POET(p=Y.shape[0], T=Y.shape[1], demeaned_matrix=Y, M=252, ic=ic1)
print('min_k: ', min_k)


# Decide k
'''
Choose the value of k: min_k or your choice
Usually the k can't be decided by the formula, because it requires a huge jump of eigenvalues.
As a result, we usually use Elbow Method.
'''
k = 10 # or min_k

# PCA
train_u, val_u = data_split(Y=Y, k=k)

# C min
C_min = find_c_min(train_u=train_u, M = 1, step=0.005)
print("Smallest C that returns True on is_positive_definite (C_min) :", C_min)

# Find C_star
C_star, min_value = find_c_star(train_u=train_u, val_u=val_u, start_point=0.055, end_point=0.065, step=0.001)
print('C_star', C_star)
print('min_value', min_value)


# Decide C star
'''
Use C_star or choose by yourself
Because it runs too slow, we use big scale then narrow it down gradually.
'''

C_star = 0.057 # or C_star
precision_matrix = compute_precision_matrix(demeaned_matrix=Y, C_star=C_star, round_decimal=7, num_eigenvalues=10)

pd.DataFrame(precision_matrix).to_csv('output/precision_matrix.csv')

