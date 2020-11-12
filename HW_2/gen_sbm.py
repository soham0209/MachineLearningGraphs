import networkx as nx
from itertools import combinations
import sys
import numpy as np
import matplotlib.pyplot as plt


def flip(prob):
    return True if np.random.random() < prob else False


if __name__ == '__main__':
    n, p, q = 200, 0.16, 0.04
    if len(sys.argv) == 4:
        n = int(sys.argv[1])
        p = float(sys.argv[2])
        q = float(sys.argv[3])
    V = np.random.permutation(n)
    lim = int(n / 2)
    v1 = set(V[0:lim])
    v2 = set(V[lim:])
    G = nx.Graph()

    G.add_nodes_from(v1, layer=0)
    G.add_nodes_from(v2, layer=1)

    for e in combinations(V, 2):
        is_edge = False
        if e[0] in v1 and e[1] in v1:
            if flip(p):
                is_edge = True
        elif e[0] in v2 and e[1] in v2:
            if flip(p):
                is_edge = True
        else:
            if flip(q):
                is_edge = True

        if is_edge:
            G.add_edge(e[0], e[1])
    if p > q:
        nx.write_gpickle(G, './sbm_graph.pkl')
    else:
        nx.write_gpickle(G, './sbm_graph_alt.pkl')

    subset_color = ["red", "blue"]
    plt.figure(figsize=(8, 8))
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    # pos = nx.bipartite_layout(G, nodes=v1, align='vertical')
    # plt.figure(figsize=(8, 8))
    nx.draw_spring(G, node_color=color, with_labels=True)
    plt.axis("equal")
    plt.show()
