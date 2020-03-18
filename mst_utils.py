######################################################################################
#
# Authors : Niharika Gauraha, Oskar Kviman
#          KTH
#          Email : {niharika, okviman}@kth.se
#
# mst_utils.py: Implements MST utility functions.
#####################################################################################


import numpy as np
import networkx as nx
from tree_utils import update_topology


def get_mst(W, t_opts):
    # W is a symmetric (2N - 1)x(2N - 1) matrix with MI entries. Last entry is link connection to root.
    G = nx.Graph()
    n_nodes = W.shape[0]
    n_nodes -= 1
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (W[i, j] == -np.infty):
                continue

            t = t_opts[i, j]  # nx.shortest_path_length(tree, i, j, weight='t')
            G.add_edge(i, j, weight=W[i, j], t=t)
    mst = nx.maximum_spanning_tree(G)
    # print("Number of edges",mst.size())
    # print("nodes", n_nodes)
    return mst


def add_midpoint_root(mst, root, n_nodes):
    n_leaves = (n_nodes + 1) // 2
    farthest_path_len = 0.
    farhest_pair = None
    for leaf_1 in range(n_leaves):
        for leaf_2 in range(leaf_1 + 1, n_leaves):
            len_ = nx.shortest_path_length(mst, leaf_1, leaf_2, weight='t')
            if len_ > farthest_path_len:
                farhest_pair = (leaf_1, leaf_2)
                farthest_path_len = len_
    path = nx.shortest_path(mst, farhest_pair[0], farhest_pair[1], weight='t')
    for i, node_i in enumerate(path):
        t_leaf2i = nx.shortest_path_length(mst, farhest_pair[0], node_i, weight='t')
        if t_leaf2i > farthest_path_len / 2:
            node_j = path[i - 1]
            #t_i2j = t_opts[node_i, node_j]/2
            t_i2j = mst[node_i][node_j]['t']
            mst.add_edge(node_i, root, t=t_i2j)
            mst.add_edge(node_j, root, t=t_i2j)
            mst.remove_edge(node_i, node_j)
            return mst


def bifurcate_mst(mst, leaves, root=0):
    neighbors = mst.adj  # dict of neighbors and connecting weights
    n_nodes = len(neighbors) + 1  # +1 for root
    D = [1 if n in leaves else 3 for n in range(n_nodes)]
    D[root] = 2
    deleted = []
    not_bifurcated = True
    while not_bifurcated:
        deleted, mst = deletion_step(mst, deleted, n_nodes, root)
        if len(deleted) == 0 and np.all([mst.degree(n) == D[n] for n in mst]):
            break
        deleted, mst = insertion_step(mst, deleted, n_nodes, root, D)
    mst = add_midpoint_root(mst, root, n_nodes)
    update_topology(mst, root)
    return mst


def deletion_step(mst, deleted, n_nodes, root):
    # deletion step (proposition 5.3 in SEM paper)
    n_leaves = (n_nodes + 1) // 2
    for j in range(n_leaves, n_nodes):
        if j in deleted or j == root:
            continue
        d = mst.degree(j)
        if d == 1:
            # internal node is leaf
            mst.remove_node(j)
            deleted.append(j)
        elif d == 2:
            nbor_i, nbor_k = [(node, mst.adj[j][node]["t"]) for node in mst.adj[j]]
            t_new = nbor_i[1] + nbor_k[1]
            mst.add_edge(nbor_i[0], nbor_k[0], t=t_new)
            mst.remove_node(j)
            deleted.append(j)
    return deleted, mst


def insertion_step(mst, deleted, n_nodes, root, D):
    # insertion step (proposition 5.4)
    eps = 1e-10  # small positive duration used in insertion step
    for i in range(n_nodes):
        if i in deleted or i == root:
            continue
        d = mst.degree(i)
        if d > D[i]:
            try:
                j = deleted.pop()
            except IndexError:
                break
            nbors_i = [(node, mst.adj[i][node]["t"]) for node in mst.adj[i]]
            if D[i] == 3:
                idx = np.argsort([nbor[1] for nbor in nbors_i])
                mst.add_edge(i, j, t=eps)
                for id in idx[:2]:
                    mst.add_edge(nbors_i[id][0], j, t=nbors_i[id][1])
                    mst.remove_edge(nbors_i[id][0], i)
            else:
                # D[i] = 1
                mst.add_edge(i, j, t=eps)
                for nbor in nbors_i:
                    mst.add_edge(nbor[0], j, t=nbor[1])
                    mst.remove_edge(nbor[0], i)
    return deleted, mst
