import numpy as np
from sklearn.metrics import accuracy_score


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


def select_al_thresholds(al_probs, dp_preds, dp_indices, labels):
    """
    Select AL threshold to maximize label precision
    :param al_probs: probs predicted by AL model
    :param dp_probs: labels predicted by DP model
    :param dp_indices: indices for predicted data
    :param labels: ground-truth labels
    :return:
    """
    al_preds = np.argmax(al_probs, axis=1)
    al_conf = np.max(al_probs, axis=1)
    candidate_thres = np.unique(al_conf)
    step = len(candidate_thres) // 100
    candidate_thres = candidate_thres[::step]
    best_precision = 0.0
    best_theta = 0.0
    for theta in candidate_thres:
        preds = - np.ones_like(labels)
        al_indices = al_conf > theta
        preds[dp_indices] = dp_preds
        preds[al_indices] = al_preds[al_indices]
        active_indices = preds != -1
        active_preds = preds[active_indices]
        active_labels = labels[active_indices]
        precision = accuracy_score(active_labels, active_preds)
        if precision > best_precision:
            best_precision = precision
            best_theta = theta

    return best_theta


def merge_filtered_probs(al_probs, dp_probs, dp_preds, dp_indices, theta):
    """
    Combine AL preds with DP preds
    :param al_probs:
    :param dp_probs:
    :param dp_preds:
    :param dp_indices:
    :param theta:
    :return: predicted_soft_labels, predicted_hard_labels, predicted_indices
    """
    al_conf = np.max(al_probs, axis=1)
    probs = np.zeros_like(al_probs)
    preds = - np.ones(len(al_probs))
    probs[dp_indices, :] = dp_probs
    preds[dp_indices] = dp_preds
    al_indices = al_conf > theta
    probs[al_indices,:] = al_probs[al_indices,:]
    preds[al_indices] = np.argmax(al_probs[al_indices,:], axis=1)
    predicted_indices = preds != -1
    probs = probs[predicted_indices,:]
    preds = preds[predicted_indices]
    return probs, preds, predicted_indices


