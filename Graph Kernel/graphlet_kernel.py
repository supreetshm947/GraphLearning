import random

import networkx as nx
import numpy as np
from random import sample

NUMBER_OF_GRAPHLET = 34
NUMBER_OF_ROUNDS = 1000
NUMBER_NODES_INDUCED_SG = 5
SEED = 999999
random.seed(SEED)


def get_graphlet(num_of_graphlet: int = NUMBER_OF_GRAPHLET):
    graphs = list()
    atlas_gp = nx.graph_atlas_g()
    while len(graphs) < num_of_graphlet:
        g = atlas_gp.pop()
        if g.number_of_nodes() == 5:
            graphs.append(g)
    return graphs


def get_embeddings(graphs):
    graphlet = get_graphlet()
    histogram = np.full((len(graphs), NUMBER_OF_GRAPHLET), fill_value=0)
    for i in range(len(graphs)):
        nodes = list(graphs[i].nodes())
        for j in range(NUMBER_OF_ROUNDS):
            induced_subgraph = graphs[i].subgraph(sample(nodes, 5))
            for k in range(len(graphlet)):
                if nx.is_isomorphic(induced_subgraph, graphlet[k]):
                    histogram[i][k] += 1
    return histogram
