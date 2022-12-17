import networkx as nx
import numpy as np
import data_utils as utils

HASH_FUNC = hash
NUM_ROUNDS = 4


def get_wl_embedding(g: nx.Graph, num_rounds: int = NUM_ROUNDS):
    recolor_node(g)
    node_labels = g.nodes(data=True)
    for r in range(num_rounds):
        for node_label in node_labels:
            node_label[1]['multiset'] = sorted([
                                          node_label[1]['color'] for x in g.neighbors(node_label[0])
                                      ] + [node_label[1]['color']])
        recolor_node(g)
    return get_node_colors(g)


def recolor_node(g: nx.Graph):
    n = g.number_of_nodes()
    node_labels = g.nodes(data=True)
    for node_label in node_labels:
        if 'multiset' in node_label[1]:
            val = abs(HASH_FUNC(str(node_label[1]['multiset'])))
        else:
            val = abs(HASH_FUNC(str(node_label[1]['node_label'])))
        node_label[1]['color'] = val


def get_node_colors(g):
    return [node[1]["color"] for node in g.nodes(data=True)]


def get_embeddings(graphs):
    distinct_color = set()
    col_graphs = list()
    for i in graphs:
        col_graphs.append(get_wl_embedding(i, 4))
        distinct_color.update(col_graphs[-1])
    embeddings = np.full((len(graphs), len(distinct_color)), fill_value=0)
    for i in range(len(col_graphs)):
        for j in col_graphs[i]:
            embeddings[i][int((j / 9999999999999999999) * len(distinct_color))] += 1
    return embeddings
