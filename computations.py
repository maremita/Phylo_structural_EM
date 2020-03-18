######################################################################################
#
# Authors : Niharika Gauraha, Oskar Kviman
#          KTH
#          Email : niharika@kth.se, okviman@kth.se
#
# computations.py: implements, for the SEM, necessary computational methods
#####################################################################################

import numpy as np
import networkx as nx


# define each letter of alphabet as a vector
nuc_vec = {'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.], 'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.]}
alphabet_size = 4  # number of letters in alphabet


# compute upward messages
def compute_up_messages(data, tree, evo_model):
    n_leaves, n_sites = data.shape
    root = len(tree) - 1

    # store up message for each node internal+external = 2n-1
    up_table = np.ones((len(tree), alphabet_size, n_sites))

    for i in range(n_leaves):
        up_table[i] = np.transpose([nuc_vec[c] for c in data[i]])

    for node in nx.dfs_postorder_nodes(tree, root):
        if not tree.nodes[node]['type'] == 'leaf':
            for child in tree.nodes[node]['children']:
                t_child = tree.nodes[child]['t']
                trans_matrix = evo_model.trans_matrix(t_child)
                temp_table = np.dot(trans_matrix, up_table[child])
                up_table[node] = np.multiply(up_table[node], temp_table)

    return up_table


# compute down messages
def compute_down_messages(data, tree, evo_model, up_table):
    n_leaves, n_sites = data.shape
    root = len(tree) - 1

    # store down message for each node intrenal+external = 2n-1
    down_table = np.ones((2 * n_leaves - 1, alphabet_size, n_sites))

    for node in nx.dfs_preorder_nodes(tree, root):
        if not node == root:
            parent = tree.nodes[node]['parent']
            if parent == root:
                parent_factor = down_table[root]
            else:
                t_parent = tree.nodes[parent]['t']
                trans_matrix = evo_model.trans_matrix(t_parent)
                parent_factor = np.dot(trans_matrix, down_table[parent])

            # compute sibling factor (there is only one sibling for a tree)
            for child in tree.nodes[parent]['children']:
                if child != node:
                    t_child = tree.nodes[child]['t']
                    trans_matrix = evo_model.trans_matrix(t_child)
                    sibling_factor = np.dot(trans_matrix, up_table[child])

            down_table[node] = np.multiply(parent_factor, sibling_factor)

    return down_table


# returns likelihood for each site as well as aggregated for all sites
def compute_loglikelihood(up_table, stat_prob):
    n_sites = up_table[0][0].size
    ll_sites = np.ones(n_sites)
    log_likelihood = 0

    for pos in range(n_sites):
        tmp = np.log(np.sum(np.multiply(up_table[-1, :, pos], stat_prob)))
        ll_sites[pos] = tmp
        log_likelihood += tmp

    return ll_sites, log_likelihood


def compute_w_ij(S_ij, trans_matrix, stat_prob, sigma=1.):
    # Computes L_{local} as in SEM paper, i.e. an entry in W matrix.
    # S_ij and trans_matrix (parameterized by t) are a 4x4 matrices. stat_prob is a vector of stationary probs.
    n_states = S_ij.shape[0]
    stat_prob = stat_prob.reshape(n_states)
    log_term = np.log(trans_matrix) - np.log(stat_prob)
    l_local = np.sum(S_ij * log_term)
    return l_local


def count_edges(node_i, tree, evo_model, up_table, down_table):
    t_ij = tree.nodes[node_i]['t']
    trans_matrix = evo_model.trans_matrix(t_ij)
    up_tmp = up_table[node_i]
    down_tmp = down_table[node_i]
    counts_edges = np.zeros((4, 4))
    for m in range(down_tmp.shape[1]):
        tmp = np.outer(up_tmp[:, m], down_tmp[:, m]) * evo_model.stat_prob[0] * trans_matrix
        counts_edges += tmp / np.sum(tmp)
    return counts_edges


def count_node(node, tree, evo_model, up_table, down_table):
    n_leaves = (len(tree) + 1) // 2
    root = len(tree) - 1
    n_sites = up_table[0][0].size
    counts_node = np.ones((alphabet_size, n_sites))

    if node < n_leaves:
        counts_node = up_table[node]
    else:
        if not node == root:
            t = tree.nodes[node]['t']
            trans_matrix = evo_model.trans_matrix(t)
        up_tmp = up_table[node]
        down_tmp = down_table[node]

        for i in range(alphabet_size):
            prob = 0
            if node == root:
                prob = 1
            else:
                for j in range(alphabet_size):
                    prob += down_tmp[j] * trans_matrix[i, j]

            prob = prob * evo_model.stat_prob[i] * up_tmp[i]

            counts_node[i, :] = prob

        counts_node = counts_node / np.sum(counts_node, axis=0)

    return counts_node


def approx_count(node_i, node_j, tree, evo_model, up_table, down_table):
    count_i = count_node(node_i, tree, evo_model, up_table, down_table)
    count_j = count_node(node_j, tree, evo_model, up_table, down_table)
    n_sites = up_table.shape[-1]
    S_approx = np.zeros((4, 4))
    for m in range(n_sites):
        S_approx += np.outer(count_i[:, m], count_j[:, m])
    return S_approx


def compute_weight_matrix(tree, evo_model, up_table, down_table, weight_mat='unconstrained'):
    n_nodes = len(tree)
    new_ts = np.zeros((n_nodes, n_nodes))
    root = len(tree) - 1

    # store message/expected count for each edge
    weight_matrix = np.zeros((n_nodes, n_nodes))

    for node1 in range(n_nodes-1):
        if weight_mat == 'constrained':
            if tree.nodes[node1]['type'] == 'leaf':
                continue  # skip the leaves
        for node2 in range(n_nodes-1):
            if node2 == root:
                continue  # skip the root

            try:
                parent = tree.nodes[node2]['parent']
            except IndexError:
                parent = None
            # collect expected counts
            if node1 == parent:
                S_ij = count_edges(node2, tree, evo_model, up_table, down_table)
            else:
                S_ij = approx_count(node1, node2, tree, evo_model, up_table, down_table)

            # Find branch lengths that maximize complete log-likelihood
            t_opt = evo_model.optimize_t(S_ij)
            trans_matrix = evo_model.trans_matrix(t_opt)

            w_ij = compute_w_ij(S_ij, trans_matrix, evo_model.stat_prob)
            new_ts[node1, node2] = t_opt
            weight_matrix[node1, node2] = w_ij

    weight_matrix[weight_matrix == 0] = -np.infty
    return weight_matrix, new_ts


def perturb_W(W, sigma_l):
    eps = np.random.normal(0, sigma_l, W.shape)
    np.fill_diagonal(eps, val=0.0)
    return W + eps


