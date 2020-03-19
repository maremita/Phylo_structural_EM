from sem import sem
import unittest
import numpy.random as npr
import computations as cmp
from data_utils import simulate_seq, get_char_data
from substitution_models import JukesCantor
from tree_utils import create_tree
from plot_utils import plot_loglikelihood


class SEMTestCase(unittest.TestCase):
    def test_sem_load_data(self):
        npr.seed(3)
        jc = JukesCantor(0.1)
        data, n_leaves, n_sites = get_char_data('../data/alignment.phylip')
        T_l, ll_vec = sem(data, jc)
        plot_loglikelihood(ll_vec)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
