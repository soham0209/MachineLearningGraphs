from myGraph import *
from Gen_graphs import flip
import numpy as np
import sys
import os


if __name__ == '__main__':
    N = int(sys.argv[1])
    k = int(sys.argv[2])
    data_dir = os.path.expanduser("cora")
    edgelist_cora = np.loadtxt(os.path.join(data_dir, "cora.cites"), delimiter='\t').astype(int)

    G_1 = Graph()
    for e in edgelist_cora:
        if e[0] != e[1]:
            G_1.add_edge(e[1], e[0])
    cc_G1 = graph_cluster_coefficient(G_1)

    del G_1

    G_2 = Graph()
    for i in range(N):
        G_2.add_node(i)
    for i in G_2.nodelist:
        for j in range(1, int(k/2)+1):
            r_node = G_2.nodelist[(i+j) % N]
            l_node = G_2.nodelist[i-j]
            G_2.add_edge(G_2.nodelist[i], r_node)
            G_2.add_edge(G_2.nodelist[i], l_node)
    print('Edges in watts-strogatz', G_2.num_edges())
    cc_avg = graph_cluster_coefficient(G_2)
    print('Before rewiring ', cc_avg)
    c_0 = 3 * (k - 2) / (4 * (k - 1))
    print('Theoretical ', c_0)
    q_range = np.arange(0, 1.05, 0.05)
    cc_diff = float('inf')
    cc_nearest = c_0
    q_best = 0
    for q in q_range:
        G_tmp = G_2.__copy__()
        edges_to_traverse = list(G_tmp.edge_list)
        for e in edges_to_traverse:
            if flip(q):
                G_tmp.remove_edge(e[0], e[1])
                new_e = np.random.randint(0, N, (2, ))
                G_tmp.add_edge(new_e[0], new_e[1])
        cc_avg = graph_cluster_coefficient(G_tmp)
        print('q:', q, 'CC: ', cc_avg)
        if abs(cc_avg - cc_G1) < cc_diff:
            cc_nearest = cc_avg
            cc_diff = abs(cc_avg - cc_G1)
            q_best = q
        del G_tmp
    print('q:', q_best, 'Nearest cluster coefficient ', cc_nearest)
    print('Clustering coefficient of G_1 is', cc_G1)
