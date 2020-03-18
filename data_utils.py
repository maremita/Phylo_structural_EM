import numpy as np
import re
import networkx as nx


# define parameters
nuc_names = ['A', 'C', 'G', 'T']
transform = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
regex = re.compile('[ACGT]')


def get_char_data(name='infile'):
    """
    Returns numerical representation of dna sequences, number of sequences (N), length of sequences (M).
    A: 0, T: 1, C: 2, G: 3
    """
    regex = re.compile('[ATCG]')
    # transform = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    sequences, N, M = parse_data(name)
    numeric_sequences = np.chararray((N, M), unicode=True)
    n = 0
    for seq in sequences:
        sites = re.findall(regex, seq)
        if len(sites) == 0 or sites is None:
            continue
        sites = sites[-M:]
        numeric_sequences[n, :] = np.array([site for site in sites])
        n += 1
    return numeric_sequences, N, M


def parse_data(name='infile'):
    f = open(name, 'r')
    sequences = []
    meta_data = f.readline().split()
    num_sequences = int(meta_data[0])
    len_sequences = int(meta_data[1])

    for i in range(len_sequences):
        seq = f.readline().strip('\n')
        sequences.append(seq)

    return sequences, num_sequences, len_sequences


# simulate sequences given the tree topology and rate matrices
def simulate_seq(tree, evo_model, ndata=10):
    n_nodes = len(tree)
    root = n_nodes - 1
    n_leaves = (n_nodes + 1) // 2
    pt_matrix = [np.zeros((4, 4)) for i in range(2 * n_leaves - 2)]

    # do postorder tree traversal to compute the transition matrices
    for node in nx.dfs_postorder_nodes(tree, root):
        if not tree.nodes[node]['type'] == 'root':
            t = tree.nodes[node]['t']
            pt_matrix[node] = evo_model.trans_matrix(t)

    simuData = []
    status = [''] * (2 * n_leaves - 1)
    for run in range(ndata):
        for node in nx.dfs_preorder_nodes(tree, root):
            if tree.nodes[node]['type'] == 'root':
                status[node] = np.random.choice(4, size=1, p=evo_model.stat_prob)[0]
            else:
                parent = tree.nodes[node]['parent']
                status[node] = np.random.choice(4, size=1, p=pt_matrix[node][status[parent]])[0]

        simuData.append([nuc_names[i] for i in status[:n_leaves]])

    return np.transpose(simuData)
