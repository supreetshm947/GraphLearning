import networkx as nx

nci1 = nx.read_gpickle("../datasets/datasets/NCI1/data.pkl")
graph = nci1[0]
node1 = graph.nodes()[1]
print(list(nx.all_simple_paths(graph, source=1, target=1, cutoff=3)))