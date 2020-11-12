import sys
import numpy as np
from itertools import combinations
from tqdm import tqdm
from myGraph import *

np.random.seed(33)


def flip(prob):
    return True if np.random.random() < prob else False


def gen_er_graph(num_nodes, p_act) -> Graph:
    g = Graph()
    nodelist = [i for i in range(num_nodes)]
    for e in combinations(nodelist, 2):
        if flip(p_act):
            g.add_edge(e[0], e[1])
    return g


def gen_er_graph_directed(num_nodes, p_act):
    g = DiGraph()
    nodelist = [i for i in range(num_nodes)]
    for e in combinations(nodelist, 2):
        if flip(p_act):
            g.add_edge(e[0], e[1])
    return g


if __name__ == '__main__':
    n = int(sys.argv[1])
    p = float(sys.argv[2])
    frac = np.log(n) / float(n)
    p_f = p * frac
    connected_count = 0
    for expt in tqdm(range(100)):
        G = gen_er_graph(n, p_f)
        if is_connected(G):
            connected_count += 1
        del G
    print('n = ', n, ',p = ', p, ',p log(n)/n = ', p_f, ',Times connected ', connected_count)
