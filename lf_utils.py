import numpy as np
from sklearn.metrics import accuracy_score
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from data_utils import filter_abstain
import torch


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


def check_filter(sampler, label_model_type, filtered_feature_indices, train_dataset, valid_dataset, use_valid_labels,
                 seed, device):
    """
    Check whether to use LF filtering or not
    :param sampler:
    :param label_model:
    :param filtered_feature_indices:
    :param valid_data:
    :param use_valid_labels:
    :return:
    """
    # calculate label precision with no LF filtering
    lfs = sampler.create_label_functions()
    if not check_all_class(lfs, train_dataset.n_class):
        return None, lfs

    L_train = train_dataset.generate_label_matrix(lfs=lfs)
    L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
    L_tr_active, y_tr_active, tr_indices = filter_abstain(L_train, train_dataset.ys)
    L_val_active, y_val_active, val_indices = filter_abstain(L_valid, valid_dataset.ys)
    if label_model_type == "mv":
        lm = MajorityLabelVoter(cardinality=train_dataset.n_class)
    elif label_model_type == "snorkel":
        if device == "cuda":
            torch.cuda.empty_cache()
        lm = LabelModel(cardinality=train_dataset.n_class, device=device)
        lm.fit(L_train=L_tr_active, Y_dev=y_val_active, seed=seed)
    else:
        raise ValueError("label model not supported.")
    if use_valid_labels:
        val_pred = lm.predict(L_val_active)
        val_precision = accuracy_score(y_val_active, val_pred)
    else:
        val_pred = lm.predict(L_train[sampler.sampled_idxs, :])
        val_labels = sampler.sampled_labels
        val_precision = accuracy_score(val_labels, val_pred)

    # calculate label precision with LF filtering
    filtered_lfs = sampler.create_label_functions(filtered_features=filtered_feature_indices)
    if not check_all_class(filtered_lfs, train_dataset.n_class):
        return lm, lfs

    L_train = train_dataset.generate_label_matrix(lfs=filtered_lfs)
    L_valid = valid_dataset.generate_label_matrix(lfs=filtered_lfs)
    L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
    L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)
    if label_model_type == "mv" or L_tr_filtered.shape[1] < 3:
        filtered_lm = MajorityLabelVoter(cardinality=train_dataset.n_class)
    elif label_model_type == "snorkel":
        filtered_lm = LabelModel(cardinality=train_dataset.n_class)
        filtered_lm.fit(L_train=L_tr_filtered, Y_dev=y_val_filtered, seed=seed)
    else:
        raise ValueError("label model not supported.")
    if use_valid_labels:
        val_pred = filtered_lm.predict(L_val_filtered)
        val_precision_filtered = accuracy_score(y_val_filtered, val_pred)
    else:
        val_pred = filtered_lm.predict(L_train[sampler.sampled_idxs, :])
        val_labels = sampler.sampled_labels
        val_precision_filtered = accuracy_score(val_labels, val_pred)

    if val_precision > val_precision_filtered:
        # keep all LFs
        return lm, lfs
    else:
        # keep filtered LFs
        return filtered_lm, filtered_lfs


def select_al_thresholds(al_probs, dp_preds, dp_indices, labels):
    """
    Select AL threshold to maximize label precision
    :param al_probs: probs predicted by AL model
    :param dp_preds: labels predicted by DP model
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


