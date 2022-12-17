import networkx as nx
import numpy as np

import data_utils

WALK_LENGTH = 10


def get_embeddings(graphs, walk_length=WALK_LENGTH):
    histogram = np.full((len(graphs), walk_length), fill_value=0)
    for i, graph in enumerate(graphs):
        walks = get_closed_walk_kernel(graph, walk_length)
        for j in range(walk_length):
            for k, walk in enumerate(walks):
                if k == walk[j]:
                    histogram[i][j] += 1
    return histogram


def get_closed_walk_kernel(graph, walk_length):
    adjacency_matrix = data_utils.get_adjacency_matrix(graph)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0))
    degree_inverse = np.linalg.inv(degree_matrix) if np.linalg.det(degree_matrix) else np.linalg.pinv(degree_matrix)
    transition_matrix = np.dot(adjacency_matrix, degree_inverse)
    walks = np.full((graph.number_of_nodes(), walk_length), fill_value=0)
    for node in range(graph.number_of_nodes()):
        p = np.full(graph.number_of_nodes(), fill_value=0)
        p[node] = 1
        walks[node][0] = node
        for i in range(walk_length - 1):
            p = p.dot(transition_matrix)
            walks[node][i + 1] = np.argmax(p)
    return walks
