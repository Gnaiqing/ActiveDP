import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from label_model import get_label_model
from data_utils import filter_abstain
from csd import StructureDiscovery
from lf_utils import check_all_class


def get_al_model(al_model_type, seed):
    if al_model_type == "logistic":
        al_model = LogisticRegression(random_state=seed)
    elif al_model_type == "decision-tree":
        al_model = DecisionTreeClassifier(random_state=seed)
    elif al_model_type is None:
        al_model = None
    else:
        raise ValueError(f"AL model {al_model_type} not supported.")

    return al_model


def get_filtered_indices(sampler, filter_method, ci_alpha=0.01):
    """
    Filter the LFs returned by the user
    :param sampler:
    :param filter_method: method used for filtering
    :param ci_alpha: alpha value for conditional independence test
    :return: a set of feature indices used for generating LFs
    """
    labeled_dataset = sampler.create_labeled_dataset()
    selected_features = np.nonzero(sampler.feature_mask)[0]
    labeled_dataset.xs = labeled_dataset.xs[:, selected_features]
    labeled_dataset.feature_names = labeled_dataset.feature_names[selected_features]
    structure_learner = StructureDiscovery(labeled_dataset)
    filtered_features = structure_learner.get_neighbor_nodes(method=filter_method,
                                                             alpha=ci_alpha,
                                                             display=False)

    filtered_feature_indices = []
    train_dataset = sampler.dataset
    for feature_name in filtered_features:
        j = train_dataset.get_feature_idx(feature_name)
        if j != -1:
            filtered_feature_indices.append(j)

    filtered_feature_indices = np.sort(filtered_feature_indices)
    return filtered_feature_indices


def check_filter(sampler, label_model_type, filtered_feature_indices, train_dataset, valid_dataset, use_valid_labels,
                 seed, device, tune_params):
    """
    Check whether to use LF filtering or not
    :param sampler: active sampler
    :param label_model_type: label model in DP
    :param filtered_feature_indices: the features used to generate LFs after filtering
    :param valid_data: validation dataset
    :param use_valid_labels: whether use validation labels or not
    :return: label_model, label_functions
    """
    # calculate label precision with no LF filtering
    lfs = sampler.create_label_functions()
    if not check_all_class(lfs, train_dataset.n_class):
        return None, lfs

    L_train = train_dataset.generate_label_matrix(lfs=lfs)
    L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
    L_tr_active, y_tr_active, tr_indices = filter_abstain(L_train, train_dataset.ys)
    L_val_active, y_val_active, val_indices = filter_abstain(L_valid, valid_dataset.ys)
    if len(lfs) < 3:
        lm = get_label_model("mv", cardinality=train_dataset.n_class)
    else:
        lm = get_label_model(label_model_type, cardinality=train_dataset.n_class)

    if label_model_type == "snorkel":
        lm.fit(L_tr_active, L_val_active, y_val_active, tune_params=tune_params)

    if use_valid_labels:
        val_pred = lm.predict(L_val_active)
        val_precision = accuracy_score(y_val_active, val_pred)
    else:
        # use the labeled subset in training set to estimate validation precision
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
    if len(filtered_lfs) < 3:
        filtered_lm = get_label_model("mv", cardinality=train_dataset.n_class)
    else:
        filtered_lm = get_label_model(label_model_type, cardinality=train_dataset.n_class)
    if label_model_type == "snorkel":
        filtered_lm.fit(L_tr_filtered, L_val_filtered, y_val_filtered, tune_params=tune_params)

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
    step = np.ceil(len(candidate_thres) / 100).astype(int)
    candidate_thres = np.append(candidate_thres[::step], 1.0)
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
