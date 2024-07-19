import numpy as np
import matplotlib.pyplot as plt

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def ic1(T, p):
    output = (p+T)/(p*T) * np.log( (p*T) / (p+T))
    return output

def ic2(T, p):
    output = (p+T)/(p*T) * np.log(min(p, T))
    return output

def plot_eigenvalues(demeaned_matrix):
    YY = np.dot(demeaned_matrix, demeaned_matrix.T)
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(YY)

    # Print eigenvalues and eigenvectors
    eigenvectors = np.dot(Y.T, eigenvectors)

    # Plot eigenvalues for the Elbow Method
    data = eigenvalues[:100]
    plt.plot(data)
    plt.title('Line Plot of the List')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def pca(matrix, n_components):
    # Demean
    matrix_mean = np.mean(matrix, axis=0)
    matrix_centered = matrix - matrix_mean

    # Covariance matrix
    covariance_matrix = np.cov(matrix, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

      # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:n_components]
    top_eigenvectors = eigenvectors[:, top_indices]

    # Projection
    projected_data = np.dot(matrix_centered, top_eigenvectors)

    return projected_data, top_eigenvectors, matrix_mean

def pca_reconstruction(projected_data, pca_components, matrix_mean):
    reconstructed_data = np.dot(projected_data, pca_components.T) + matrix_mean

    return reconstructed_data


def theta_ij(u, p, T):
    theta = np.zeros(u.shape)

    for i in range(p):
        for j in range(T):
            sigma = (1/T) *sum( \
                [u[i,t] * u[j, t] for t in range(T)] \
                )
            theta[i, j] = (1 / T) * sum( \
                [(u[i, t] * u[j, t] - sigma)**(2) for t in range(T)] \
                )

    return theta # matrix

def indicator_function(condition):
    """
    Simple indicator function:
    Returns 1 if the condition is True, otherwise returns 0.
    """
    if condition:
        return 1
    else:
        return 0

def get_W_T(p, T):
    out = 1/(p**(1/2)) + (np.log(p)/T)**(1/2)
    return out

def tau_ij(C, W_T, theta_ij_hat):
    out = C * W_T * ((theta_ij_hat)**(1/2))
    return out

def s_ij(value, threshold): # hard threshold
    out = value * (indicator_function(np.abs(value) >= threshold))
    return out

def get_r_tau_ij(matrix, tau_ij):
    result_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                result_matrix[i, j] = matrix[i, j]
            else:
                result_matrix[i, j] = s_ij(matrix[i, j], tau_ij[i, j]) * (indicator_function(np.abs(matrix[i, j]) >= tau_ij[i, j]))
    return result_matrix

# sigma J2
def get_sigma_j2(u, T_j2):
    p = u.shape[0]
    T = u.shape[1]
    sigma = np.zeros((p, p))

    for t in range(T_j2):
        u_t = u[:, t].reshape(-1, 1)
        sigma += np.dot(u_t, u_t.T)

    return sigma


# def is_positive_definite(matrix):
#     eigenvalues = np.linalg.eigvals(matrix)
#     return np.all(eigenvalues > 0)

def make_positive_definite(matrix, epsilon=1e-8):
    if not is_positive_definite(matrix):
        n = matrix.shape[0]
        min_eigenvalue = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix += np.eye(n) * (-min_eigenvalue + epsilon)
    return matrix