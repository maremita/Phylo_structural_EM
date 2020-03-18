import matplotlib.pyplot as plt


def plot_loglikelihood(ll_vec, true_ll=None):
    plt.plot(range(len(ll_vec)), ll_vec)
    if true_ll is not None:
        plt.hlines(true_ll, 0, len(ll_vec), colors='red', linestyles='solid')
    plt.ylabel('Log-likelihood')
    plt.show()
