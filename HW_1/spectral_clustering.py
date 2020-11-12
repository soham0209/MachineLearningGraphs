from myGraph import *
import os
import numpy as np
from tqdm import tqdm


def n_cut(cut, s, v_s):
    vol_s = 0
    vol_v_s = 0
    for d in s:
        vol_s += len(G_4.nodes[d].neighbors)
    for d in v_s:
        vol_v_s += len(G_4.nodes[d].neighbors)
    return float(cut / vol_s) + float(cut / vol_v_s)


def r_cut(cut, s, v_s):
    return float(cut / len(s)) + float(cut / len(v_s))


def calculate_cut(s: set, v_s: set, metric):
    cut = 0
    for ed in G_4.edge_list:
        if ed[0] in s and ed[1] in v_s:
            cut += 1
        elif ed[1] in s and ed[0] in v_s:
            cut += 1
    if metric == 'ncut':
        return n_cut(cut, s, v_s)
    elif metric == 'rcut':
        return r_cut(cut, s, v_s)
    else:
        raise Exception('Metric should be n-cut or r-cut')


def sweep_cut(nodes, metric='ncut'):
    min_conductance = float('inf')
    k = -1
    for c in tqdm(range(1, len(nodes) - 1)):
        s = set(nodes[:c])
        v_s = set(nodes[c:])
        conductance = calculate_cut(s, v_s, metric)
        # print('k: ', c, 'nCut: ', conductance)
        if conductance < min_conductance:
            min_conductance = conductance
            k = c
    return k, min_conductance


if __name__ == '__main__':
    data_dir = os.path.expanduser("./cora")
    edgelist_cora = np.loadtxt(os.path.join(data_dir, "cora.cites"), delimiter='\t').astype(int)
    G_1 = Graph()
    for e in edgelist_cora:
        if e[0] != e[1]:
            G_1.add_edge(e[1], e[0])
    # Graph built
    connected_component_dict = dict()
    visited = dict()
    max_comp, max_start = float('-inf'), float('-inf')
    for i, n in enumerate(G_1.nodelist):
        if n not in visited:
            comp = bfs_visit(G_1, n, visited)
            connected_component_dict[n] = [k for k in comp]
            comp_len = len(connected_component_dict[n])
            if comp_len > max_comp:
                max_comp = comp_len
                max_start = n
    print('G_4 size:', max_comp)
    connected_component = connected_component_dict[max_start]
    G_4 = Graph()
    del visited
    del connected_component_dict
    subgraph_node_map = dict()
    subgraph_nodes = []
    for i, n in enumerate(connected_component):
        subgraph_node_map[n] = i
        subgraph_nodes.append(n)
        G_4.add_node(i)
    num_subgraph_nodes = len(subgraph_node_map)
    A = np.zeros((num_subgraph_nodes, num_subgraph_nodes), dtype=int)
    D = np.zeros((num_subgraph_nodes, num_subgraph_nodes), dtype=int)
    for n in subgraph_node_map:
        nbrs = G_1.nodes[n].neighbors
        u_ = subgraph_node_map[n]
        deg = len(nbrs)
        for v in nbrs:
            v_ = subgraph_node_map[v]
            G_4.add_edge(u_, v_)
            A[u_][v_] = 1
        D[u_][u_] = deg
    # Laplacian Matrix
    L = D - A
    DH = np.nan_to_num(np.divide(np.sqrt(D), D))
    L_sym = np.identity(num_subgraph_nodes, dtype=float) - np.matmul(np.matmul(DH, A), DH)
    print('Laplacian and normalized laplacian computed')
    # print(L_sym)
    print('Computing Eigen values')
    w, v = np.linalg.eig(L)
    wsym, vsym = np.linalg.eig(L_sym)
    print('Eigen values computed')
    xind = np.argsort(w)
    x_sym_ind = np.argsort(wsym)
    x = v[:, xind[1]]
    xsym = vsym[:, x_sym_ind[1]]
    x_ind = np.argsort(x)
    k, val = sweep_cut(x_ind.flatten(), 'ncut')
    print('Cut found at k =', k)
    # print('V/s = ', subgraph_nodes[k:])
    print('N-cut(s) = ', val)

    x_sym_ind = np.argsort(np.matmul(DH, xsym))
    k, val = sweep_cut(x_sym_ind.flatten(), 'ncut')
    print('Cut found at k = ', k)
    # print('V/s = ', subgraph_nodes[k:])
    print('N-cut(s`) = ', val)
    k, val = sweep_cut(x_ind.flatten(), 'rcut')
    print('Cut found at k = ', k)
    # print('V/s = ', subgraph_nodes[k:])
    print('R-cut(s) = ', val)

    k, val = sweep_cut(x_sym_ind.flatten(), 'rcut')
    print('Cut found at k = ', k)
    # print('V/s = ', subgraph_nodes[k:])
    print('R-cut(s`) = ', val)
    # x_nodes = [subgraph_nodes[i] for i in x_ind]
