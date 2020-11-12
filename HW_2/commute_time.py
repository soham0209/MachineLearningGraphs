import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm


def K_uv(s, t, f: np.array, d, eigvalues):
    f_u = f[s, 1:] / np.sqrt(d[s])
    f_v = f[t, 1:] / np.sqrt(d[t])
    f_temp = (f_u - f_v) * (f_u - f_v)
    f_temp = f_temp / (1 - eigvalues[1:])
    return vol * np.sum(f_temp)


def H_uv(s, t, f: np.array, d, eigvalues):
    f_u = f[s, 1:] / np.sqrt(d[s])
    f_v = f[t, 1:] / np.sqrt(d[t])
    f_temp = f_u * (f_u - f_v)
    f_temp = f_temp / (1 - eigvalues[1:])
    return vol * np.sum(f_temp)


def make_orthonormal(eigenvectors):
    normed_eigvecs = np.zeros(eigenvectors.shape)
    for j in range(eigenvectors.shape[1]):
        normed_eigvecs[:, j] = eigenvectors[:, j] / np.linalg.norm(eigenvectors[:, j])
    return normed_eigvecs


if __name__ == '__main__':
    G: nx.Graph = nx.read_gpickle('./sbm_graph.pkl')
    v1 = set()
    v2 = set()
    for n in G.nodes:
        if G.nodes[n]['layer'] == 0:
            v1.add(n)
        else:
            v2.add(n)
    A = nx.adj_matrix(G).toarray()
    D = np.sum(A, axis=1)
    vol = np.sum(D)
    DH = 1 / np.sqrt(D)
    DH = np.diag(DH)
    N = DH @ A @ DH
    num_nodes = G.number_of_nodes()
    lim = int(num_nodes/2)
    # DH = np.diag(np.divide(np.sqrt(D), D))
    # L_sym = np.identity(G.number_of_nodes(), dtype=float) - np.matmul(np.matmul(DH, A), DH)
    eigval, eigvec = np.linalg.eig(N)
    eigvec = make_orthonormal(eigvec)
    nodelist = [i for i in range(num_nodes)]
    # K = np.zeros((num_nodes, num_nodes))
    in_com = []
    across_com = []
    hit_times = []
    for (u, v) in combinations(nodelist, 2):
        com_time = K_uv(u, v, eigvec, D, eigval)
        if u in v1 and v in v1:
            in_com.append(com_time)
        elif u in v2 and v in v2:
            in_com.append(com_time)
        else:
            across_com.append(com_time)
    print('In block mean:', np.mean(in_com))
    print('In block var:', np.var(in_com))
    # num_bins = np.sqrt(num_nodes).astype('int')
    num_bins = 100
    fig = plt.figure()
    plt.hist(in_com, num_bins)
    plt.savefig('./inblock.png')
    plt.close(fig)

    print('Across block mean:', np.mean(across_com))
    print('Across block var:', np.var(across_com))
    fig = plt.figure()
    plt.hist(across_com, num_bins)
    plt.savefig('./acrossblock.png')
    plt.close(fig)
