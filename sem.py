#############################################################################
#
# Authors: Oskar Kviman & Niharika Gauraha
#          KTH
#          E-mail: {okviman, niharika}@kth.se
#
# sem.py: main file for running the Structural EM algorithm
#############################################################################

import numpy as np
from tree_utils import create_tree
from substitution_models import JukesCantor
from data_utils import get_char_data, simulate_seq
import computations as cmp
from mst_utils import get_mst, bifurcate_mst
import plot_utils as plot


def sem(data, evol_mod, perturb=True):
    n_leaves, n_sites = data.shape

    # create tree
    T_0 = create_tree(n_leaves)  # init tree topology
    leaves = [n for n in T_0 if T_0.nodes[n]['type'] == 'leaf']
    root = len(T_0) - 1

    # annealing variables
    sigma_l = 0.1  # annealing std
    rho = 0.95  # cooling factor

    # EM
    max_iter = 100
    T_l = T_0  # init topology
    ll_vec = []  # log-likelihood
    for iter in range(max_iter):
        # E-step
        up_table = cmp.compute_up_messages(data, T_l, evol_mod)
        down_table = cmp.compute_down_messages(data, T_l, evol_mod, up_table)

        # compute log-likelihood
        ll_sites, ll = cmp.compute_loglikelihood(up_table, evol_mod.stat_prob)
        ll_vec.append(ll)

        # M-step
        W, t_opts = cmp.compute_weight_matrix(T_l, evol_mod, up_table, down_table)
        if perturb:
            W_tilde = cmp.perturb_W(W, sigma_l)
        else:
            W_tilde = W
        mst = get_mst(W_tilde, t_opts)
        T_l = bifurcate_mst(mst, leaves, root)

        # update annealing temperature
        sigma_l *= rho
        if sigma_l <= 5e-3:
            break
        elif iter > 0 and np.abs(ll_vec[iter] - ll_vec[iter - 1]) < 1e-3:
            break
    return T_l, ll_vec


def main(load_data=False):
    # init substitution model
    jc = JukesCantor(alpha=0.1)
    true_ll = None

    if load_data:
        data, _, _ = get_char_data('data/alignment.phylip')
    else:
        n_leaves = 10
        tree = create_tree(n_leaves)
        data = simulate_seq(tree, jc, 100)
        up_table = cmp.compute_up_messages(data, tree, jc)
        ll_sites, true_ll = cmp.compute_loglikelihood(up_table, jc.stat_prob)

    T_l, ll_vec = sem(data, jc, False)
    plot.plot_loglikelihood(ll_vec, true_ll)


if __name__ == '__main__':
    main()
