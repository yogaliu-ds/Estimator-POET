import numpy as np

def compute_log_returns(sdf):
    ndt = np.array(sdf)
    log_returns = 100 * (np.log(ndt[1:, :]) - np.log(ndt[:-1, :]))
    return log_returns

def get_training_data(log_returns, training_ratio=0.8):
    T0 = int(log_returns.shape[0] * training_ratio)
    training_data = log_returns.T[:, :T0]
    return training_data, T0

def compute_cov_matrix(training_data):
    cov_matrix = np.cov(training_data.T, rowvar=False)
    return cov_matrix

def demean_matrix(matrix):
    column_means = np.mean(matrix, axis=1).reshape(-1, 1)
    demeaned_matrix = matrix - column_means
    return demeaned_matrix

# All procedure
def process_data(sdf, training_ratio=0.8):
    log_returns = compute_log_returns(sdf)
    training_data, T0 = get_training_data(log_returns, training_ratio)
    cov_matrix = compute_cov_matrix(training_data)
    demeaned_matrix = demean_matrix(training_data)
    return demeaned_matrix, cov_matrix, T0
