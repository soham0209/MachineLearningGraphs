import networkx as nx
import numpy as np
import sys


if __name__ == '__main__':
    G: nx.Graph = nx.read_gpickle('./sbm_graph.pkl')
    if len(sys.argv) < 3:
        print('python3 ppr.py <alpha> <K>')
        sys.exit('Insufficient arguments')
    alpha = float(sys.argv[1])
    k = int(float(sys.argv[2]))
    A = nx.adj_matrix(G).toarray()
    D = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / D)
    vol = np.sum(D)
    num_nodes = G.number_of_nodes()
    P = D_inv @ A
    P = np.transpose(P)
    v1 = [n for n in G.nodes if G.nodes[n]['layer'] == 0]
    v2 = [n for n in G.nodes if G.nodes[n]['layer'] == 1]
    v1 = set(v1)
    recalls = []
    for ii in range(20):
        vs = np.random.permutation(list(v1))[0:50]
        s = np.zeros((num_nodes, ))
        s[vs] = 0.05
        # s = s.reshape((s.shape[0], 1))
        v = s.reshape((s.shape[0], 1))
        M_hat = (alpha * P + ((1 - alpha)/num_nodes))
        for ki in range(k):
            v = M_hat @ v
        v = v.reshape((v.shape[0], ))
        rank_ind = np.argsort(v)[100:]
        recall_top100 = 0
        for ind in rank_ind:
            if ind in v1:
                recall_top100 += 1
        recall_top100 /= 100
        # print('Run ', ii + 1, ':', recall_top100)
        recalls.append(recall_top100)
    avg_recall = np.mean(recalls)
    print(alpha, k, avg_recall)



