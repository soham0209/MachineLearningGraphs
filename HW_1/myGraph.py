from typing import List, Dict, Set


class Node:
    def __init__(self, u):
        self.u = u
        self.neighbors: Set[int] = set()

    def __eq__(self, other):
        return self.u == other.u

    def __hash__(self):
        return hash(self.u)


class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = dict()
        self.nodelist: List[int] = []
        self.edge_list = set()
        self.is_undirected = True

    def add_node(self, u):
        if u not in self.nodes:
            self.nodes[u] = Node(u)
            self.nodelist.append(u)

    def add_edge(self, u, v):
        x, y = tuple(sorted([u, v]))
        self.add_node(u)
        self.add_node(v)
        self.nodes[u].neighbors.add(v)
        self.nodes[v].neighbors.add(u)
        self.edge_list.add((x, y))

    def remove_edge(self, u, v):
        x, y = tuple(sorted([u, v]))
        self.edge_list.remove((x, y))
        self.nodes[u].neighbors.remove(v)
        self.nodes[v].neighbors.remove(u)

    def num_nodes(self):
        return len(self.nodelist)

    def num_edges(self):
        return len(self.edge_list)

    def has_edge(self, u, v):
        return (u, v) in self.edge_list or (v, u) in self.edge_list

    def __copy__(self):
        g_tmp = Graph()
        g_tmp.nodes = dict()
        for e in self.edge_list:
            g_tmp.add_edge(e[0], e[1])
        return g_tmp


class DiGraph(Graph):
    def __init__(self):
        super().__init__()
        self.is_undirected = False

    def add_edge(self, source, target):
        self.add_node(source)
        self.add_node(target)
        self.nodes[source].neighbors.add(target)
        self.edge_list.add((source, target))

    def remove_edge(self, source, target):
        self.nodes[source].neighbors.remove(target)
        self.edge_list.remove((source, target))


def bfs(graph: Graph, s: int, visited: dict):
    q: List[int] = list()
    q.append(s)
    while q:
        u = q.pop()
        visited[u] = True
        for v in graph.nodes[u].neighbors:
            if v not in visited:
                q.append(v)


def bfs_visit(graph: Graph, s: int, visited: dict) -> set:
    connected = set()
    q: List[int] = list()
    q.append(s)
    while q:
        u = q.pop()
        visited[u] = True
        connected.add(u)
        for v in graph.nodes[u].neighbors:
            if v not in visited:
                q.append(v)
    return connected


def is_connected(graph: Graph) -> bool:
    visited: Dict[Node, bool] = dict()
    if not graph.nodelist:
        raise Exception('Graph should not be empty')
    s = graph.nodelist[0]
    bfs(graph, s, visited)
    if len(visited) == len(graph.nodelist):
        return True
    return False


def cluster_coefficient(graph: Graph, u: int):
    nbrs_u = set([n for n in graph.nodes[u].neighbors if n != u])
    num_links = 0
    for v in nbrs_u:
        if u == v:
            continue
        node_v = graph.nodes[v]
        node_v_nbrs = node_v.neighbors
        for nbr_v in node_v_nbrs:
            if nbr_v in nbrs_u:
                num_links += 1
    k = len(nbrs_u)
    if k < 2:
        return 0
    cc = (2.0 * float(num_links)) / float((k * (k - 1)))
    if graph.is_undirected:
        cc = cc / 2.0
    return cc


def graph_cluster_coefficient(graph: Graph):
    cc_avg = 0.0
    for node in graph.nodelist:
        cc_i = cluster_coefficient(graph, node)
        cc_avg += cc_i
    cc_avg = cc_avg / float(graph.num_nodes())
    return cc_avg
