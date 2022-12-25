import numpy as np
import data_utils as utils


def get_normalized_adjacency_matrix(graph):
    adjacency_matrix = utils.get_adjacency_matrix(graph)
    d_sqrt = np.sqrt(np.diag(np.sum(adjacency_matrix, axis=1)))
    d_sqrt_inverse = np.linalg.inv(d_sqrt) if np.linalg.det(d_sqrt) else np.linalg.pinv(d_sqrt)
    return np.matmul(np.matmul(d_sqrt_inverse, adjacency_matrix), d_sqrt_inverse)