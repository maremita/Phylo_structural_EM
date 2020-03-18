######################################################################################
#
# Author : Oskar Kviman, Niharika Gauraha
#          KTH
#          Email : {okviman, niharika}@kth.se
#
# substitution_models.py: collection of substitution (or evolution) models
#####################################################################################
import numpy as np


class JukesCantor:
    def __init__(self, alpha):
        self.alpha = alpha
        self.D, self.U, self.U_inv, self.Q = JC_param(alpha)
        self.stat_prob = np.array([0.25]*4)

    def trans_matrix(self, t):
        return np.dot(self.U * np.exp(self.D * t), self.U_inv)

    def optimize_t(self, S_ij):
        # optimizes branch length for maximum likelihood estimate. See notes for details

        max_branch_length = 15  # maximum separating branch length according to SEM code

        # First sum all counts where no state transitions occur, i.e. across the diagonal
        S_same = np.trace(S_ij)

        # Next sum counts where transitions have taken place
        np.fill_diagonal(S_ij, val=0.0)  # trick setting all non-transition counts to zero
        S_diff = np.sum(S_ij)

        if S_diff == 0 and S_same == 0:
            raise ValueError('Both S_same and S_diff are zero')
        elif S_diff == 0:
            return (4 * self.alpha) * 1e-10
        elif 3 * S_same - S_diff <= 0:
            return max_branch_length

        # mock variable
        u_bar = (3 * S_same - S_diff) / (3 * S_same + 3 * S_diff)

        # exact optimization
        t_opt = - np.log(u_bar) / (4 * self.alpha)
        return np.maximum(np.minimum(t_opt, 15), (4 * self.alpha) * 1e-10)


def JC_param(alpha=.1):
    # JC specific parameters from matrix decomposition
    rate_matrix = alpha * np.ones((4, 4))

    for i in range(4):
        rate_matrix[i, i] = -3 * alpha

    D, U = np.linalg.eig(rate_matrix)
    U_inv = np.linalg.inv(U)

    return D, U, U_inv, rate_matrix


