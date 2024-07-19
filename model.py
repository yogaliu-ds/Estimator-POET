import numpy as np
from utils import theta_ij, get_W_T, tau_ij, get_r_tau_ij, is_positive_definite, get_sigma_j2, pca, pca_reconstruction

def estimator_POET(p, T, demeaned_matrix, M, ic):
    Y = demeaned_matrix
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(Y, Y.T))
    eigenvectors = np.dot(Y.T, eigenvectors)
    min_K = float('inf')
    min_value = float('inf')

    for K1 in range(1, M+1):
        FK = (T**(1/2)) * eigenvectors[:, :K1] # K1_eigenvector

        # computation
        e1 = (1/T) * Y
        e1 = np.dot(e1, FK)
        e1 = np.dot(e1, FK.T) # YFF
        F2 = np.linalg.norm( Y - e1, 'fro')
        element_1 = np.log((1/(p*T)) * F2)
        element_2 = K1 * ic(T, p)
        value = element_1 + element_2
        if value < min_value:
            min_value = value
            min_K = K1
    return min_K

def data_split(Y, k):
    projected_data, top_eigenvectors, matrix_mean = pca(Y, n_components=k)
    reconstructed_data = pca_reconstruction(projected_data, top_eigenvectors, matrix_mean)  # PCA reconstruction
    u_it_hat = Y - reconstructed_data  # u_it = reconstruction_error

    # Data split
    T = u_it_hat.shape[1]
    T_j1 = int(T * (1- np.log(T)**(-1)))
    T_j2 = T - T_j1
    train_u = u_it_hat[:, :T_j1]
    val_u = u_it_hat[:, :T_j2]
    return train_u, val_u

def find_c_min(train_u, M = 1, step=0.005):
    C_values = np.arange(0, M, step)
    p = train_u.shape[0]
    T = train_u.shape[1]

    matrix = np.dot(train_u, train_u.T)

    C_min = None
    for C in C_values:
        theta = theta_ij(matrix, p, matrix.shape[0])
        W_T = get_W_T(p, T)
        tau = tau_ij(C=C, W_T=W_T, theta_ij_hat=theta)
        r_tau_ij = get_r_tau_ij(matrix, tau)

        if is_positive_definite(r_tau_ij):
            C_min = C
            break
    return C_min

def find_c_star(train_u, val_u, start_point=0.055, end_point=0.065, step=0.001):
    sigma_j2 = get_sigma_j2(val_u, 5)
    # sigma J1
    p = train_u.shape[0]
    T = train_u.shape[1]

    C_values = np.arange(start_point, end_point, step)

    matrix = np.dot(train_u, train_u.T)

    C_star = float('inf')
    min_value = float('inf')
    for C in C_values:
        # element J1
        theta = theta_ij(matrix, p, matrix.shape[0])
        W_T = get_W_T(p, T)
        tau = tau_ij(C=C, W_T=W_T, theta_ij_hat=theta)
        r_tau_ij = get_r_tau_ij(matrix, tau)
        sigma_j1 = r_tau_ij
        # print(sigma_j1)

        F2 = np.linalg.norm(sigma_j1 + sigma_j2, 'fro')
        value = F2
        if value < min_value:
            min_value = value
            C_star = C
    return C_star, min_value

def compute_precision_matrix(demeaned_matrix, C_star, round_decimal=7, num_eigenvalues=10):
    Y = demeaned_matrix
    YY = (1/331) * np.dot(Y, Y.T)
    YY =  np.round(YY, decimals=round_decimal)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(YY)

    K = num_eigenvalues
    p = eigenvectors.shape[0]
    T = eigenvectors.shape[1]
    element1 = sum([ \
        np.dot(eigenvalues[k]*(eigenvectors[:, k].reshape(-1, 1)), (eigenvectors[:, k].reshape(1, -1))) for k in range(K) \
        ])
    element2 = sum([ \
        np.dot(eigenvalues[k]*(eigenvectors[:, k].reshape(-1, 1)), (eigenvectors[:, k].reshape(1, -1))) for k in range(K, p) \
        ])

    theta = theta_ij(element2, p, T)
    W_T = get_W_T(p, T)
    tau = tau_ij(C=C_star, W_T=W_T, theta_ij_hat=theta)
    r_tau_ij = get_r_tau_ij(element2, tau)
    estimated_cov_matrix = element1 + r_tau_ij

    # cov_matrix = make_positive_definite(rounded_cov_matrix)
    rounded_cov_matrix = np.round(estimated_cov_matrix, decimals=round_decimal)

    # Precision Matrix
    precision_matrix = np.linalg.inv(rounded_cov_matrix)

    return precision_matrix