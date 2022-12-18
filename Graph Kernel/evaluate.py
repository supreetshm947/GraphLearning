from sklearn import svm
from sklearn.model_selection import cross_val_score
import networkx as nx
import data_utils as utils
import graphlet_kernel as gk
import wl_kernel as wk
import closed_walk_kernel as ck

MIN_NODES = 5


def read_data():
    nci1 = nx.read_gpickle("./datasets/datasets/NCI1/data.pkl")
    dd = nx.read_gpickle("./datasets/datasets/DD/data.pkl")
    enzymes = nx.read_gpickle("./datasets/datasets/ENZYMES/data.pkl")
    return dict({"nci1": nci1, "dd": dd, "enzymes": enzymes})


def get_labels(data: dict):
    labels = dict()
    for key in data.keys():
        lb = list()
        for graph in data.get(key):
            lb.append(utils.get_graph_label(graph))
        labels.update({key: lb})
    return labels


def filter_smaller_graphs(data, labels, min_nodes=MIN_NODES):
    for key in data:
        idx_to_del = list()
        key_data = data.get(key)
        label = labels.get(key)
        for idx, i in enumerate(key_data):
            if i.number_of_nodes() < min_nodes:
                idx_to_del.append(idx)
        for idx in idx_to_del:
            del key_data[idx]
            del label[idx]


def compute_gram_matrix(embeddings):
    return embeddings.dot(embeddings.T)


def run_eval():
    data = read_data()
    labels = get_labels(data)
    clf = svm.SVC(kernel="precomputed")
    kernels = {"Closed Walk Kernel": ck, "WL Kernel": wk}
    model_data(clf, data, kernels, labels)
    filter_smaller_graphs(data, labels)
    kernels = {"Graphlet Kernel": gk}
    model_data(clf, data, kernels, labels)


def model_data(clf, data, kernels, labels):
    for kernel in kernels.keys():
        for key in data.keys():
            graphs = data.get(key)
            target = labels.get(key)
            embeddings = kernels.get(kernel).get_embeddings(graphs)
            scores = cross_val_score(clf, compute_gram_matrix(embeddings), target, cv=10) * 100
            print("Scores for " + kernel + " for data set " + key + ":", scores)


if __name__ == '__main__':
    run_eval()
