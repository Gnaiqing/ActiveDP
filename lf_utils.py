import numpy as np


def get_lf_stats(L, labels):
    """
    Compute summary statistics of LFs
    :param L: label matrix
    :param labels: ground truth labels
    :return: dictionary containing LF stats
    """
    labels = np.array(labels)
    lf_acc = np.sum(L == labels.reshape(-1,1), axis=0) / np.sum(L != -1, axis=0)
    lf_cov = np.sum(L != -1, axis=0) / len(L)
    lf_acc_avg = np.mean(lf_acc)
    lf_acc_std = np.std(lf_acc)
    lf_cov_avg = np.mean(lf_cov)
    lf_cov_std = np.std(lf_cov)
    lf_num = L.shape[1]
    stats = {
        "lf_acc_avg": lf_acc_avg,
        "lf_acc_std": lf_acc_std,
        "lf_cov_avg": lf_cov_avg,
        "lf_cov_std": lf_cov_std,
        "lf_num": lf_num
    }
    return stats


def check_all_class(lfs, n_class):
    class_exist = np.repeat(False, n_class)
    for (j, op, v, l) in lfs:
        class_exist[l] = True

    return np.all(class_exist)





