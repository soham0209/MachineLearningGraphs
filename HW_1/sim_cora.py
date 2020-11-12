from myGraph import *
from Gen_graphs import gen_er_graph
import sys
import os
import numpy as np


if __name__ == '__main__':
    num_nodes = 2708
    p = float(sys.argv[1])
    G_2 = gen_er_graph(num_nodes, p)

    cc_avg = 0.0

    data_dir = os.path.expanduser("./cora")
    edgelist_cora = np.loadtxt(os.path.join(data_dir, "cora.cites"), delimiter='\t').astype(int)
    G_1 = Graph()

    num_self_loops = 0
    for e in edgelist_cora:
        if e[0] != e[1]:
            G_1.add_edge(e[1], e[0])

    print('Number of edges in G_1 is', len(G_1.edge_list))

    cc_avg = graph_cluster_coefficient(G_1)
    print('G_1 Clustering coeff: ', cc_avg)

    print('Number of edges in G_2 is', len(G_2.edge_list))
    cc_avg = graph_cluster_coefficient(G_2)
    print('G_2 Clustering coeff: ', cc_avg)
