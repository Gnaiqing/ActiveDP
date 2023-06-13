import numpy as np
import pandas as pd
from data_utils import AbstractDataset, TextDataset, DiscreteDataset
from scipy.stats import entropy
import abc


def get_sampler(sampler_type, dataset, seed, explore_method, exploit_method, al_feature, uncertain_type, replace=True):
    if sampler_type == "passive":
        return PassiveSampler(dataset, seed, replace=replace)
    elif sampler_type == "uncertain":
        return UncertainSampler(dataset, al_feature, uncertain_type, seed, replace=replace)
    elif sampler_type == "two-stage":
        return TwoStageSampler(dataset, explore_method, exploit_method, al_feature, uncertain_type, seed, replace=replace)
    else:
        raise ValueError(f"Sampler {sampler_type} not supported.")


class Sampler(abc.ABC):
    def __init__(self, dataset, seed, replace=True):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.replace = replace
        self.sampled_idxs = list()
        self.sampled_labels = list()
        self.sampled_features = list()
        self.feature_mask = np.repeat(False, self.dataset.n_features())  # record whether a feature is selected.

    @abc.abstractmethod
    def sample(self):
        """
        Randomly sample a data point and return its idx
        :param replace: whether sample with replacement
        :return:
        """
        pass

    def update_feedback(self, idx, label, features=None):
        assert idx == self.sampled_idxs[-1]
        self.sampled_labels.append(label)
        if features is not None:
            self.sampled_features.append(features)
            for j in features:
                self.feature_mask[j] = True

    def create_label_functions(self, filtered_features=None):
        lfs = []
        for idx, label, features in zip(self.sampled_idxs, self.sampled_labels, self.sampled_features):
            if filtered_features is not None:
                features = [f for f in features if f in filtered_features]

            for f in features:
                v = self.dataset.xs[idx, f]
                lf = (f, "=", v, label)  # (feature_idx, op, value, label)
                if lf not in lfs:
                    lfs.append(lf)

        return lfs

    def create_labeled_dataset(self, features="all", drop_const_columns=False):
        """
        Create labeled dataset using expert annotation
        :param features: "selected" or "all"
        :param drop_const_columns: whether remove columns with constant values
        :return: labeled dataset
        """
        if features == "selected":
            features = np.arange(self.dataset.n_features())[self.feature_mask]
        elif features == "all":
            features = np.arange(self.dataset.n_features())

        subset_idxs = np.array(self.sampled_idxs)
        subset_labels = np.array(self.sampled_labels)
        labeled_dataset = self.dataset.create_subset(subset_idxs, features, labels=subset_labels,
                                                     drop_const_columns=drop_const_columns)
        return labeled_dataset


class PassiveSampler(Sampler):
    def sample(self):
        is_candidate = np.repeat(True, len(self.dataset))
        if not self.replace:
            is_candidate[np.array(self.sampled_idxs)] = False

        candidates = np.arange(len(self.dataset))[is_candidate]
        if len(candidates) == 0:
            # no remaining data for selection
            return -1
        else:
            idx = self.rng.choice(candidates)
            self.sampled_idxs.append(idx)
            return idx


def uncertain_sample(dataset, candidates, al_model, al_feature, uncertain_type="entropy"):
    if al_feature == "tfidf":
        train_feature = dataset.xs_feature
    else:
        train_feature = dataset.xs

    train_probs = al_model.predict_proba(train_feature)
    candidate_train_probs = train_probs[candidates, :]
    if uncertain_type == "entropy":
        uncertain_score = entropy(candidate_train_probs, axis=1)
        idx_in_list = np.argmax(uncertain_score)
    elif uncertain_type == "margin":
        candidate_train_probs = np.sort(candidate_train_probs, axis=1)
        uncertain_score = candidate_train_probs[:, -1] - candidate_train_probs[:, -2]
        idx_in_list = np.argmax(uncertain_score)
    elif uncertain_type == "confidence":
        uncertain_score = 1 - np.max(candidate_train_probs, axis=1)
        idx_in_list = np.argmax(uncertain_score)
    else:
        raise ValueError("Uncertainty type not supported.")

    idx = candidates[idx_in_list]
    return idx


class UncertainSampler(Sampler):
    """
    Sample based on uncertainty of AL model
    """
    def __init__(self, dataset, al_feature, uncertain_type, seed, replace=True):
        super(UncertainSampler, self).__init__(dataset, seed, replace=replace)
        self.al_feature = al_feature
        self.uncertain_type = uncertain_type

    def sample(self, al_model=None):
        is_candidate = np.repeat(True, len(self.dataset))
        if self.replace and len(self.sampled_idxs) > 0:
            is_candidate[np.array(self.sampled_idxs)] = False

        candidates = np.arange(len(self.dataset))[is_candidate]
        if len(candidates) == 0:
            # no remaining data for selection
            return -1
        else:
            if al_model is None:
                idx = self.rng.choice(candidates)
            else:
                idx = uncertain_sample(self.dataset, candidates, al_model, self.al_feature, uncertain_type=self.uncertain_type)
            self.sampled_idxs.append(idx)
            return idx


class TwoStageSampler(Sampler):
    """
    Sample based on exploration(for LF discovery) or exploitation (for active learning) mode.
    """
    def __init__(self, dataset, explore_method, exploit_method, al_feature, uncertain_type, seed, replace=True):
        super(TwoStageSampler, self).__init__(dataset, seed, replace=replace)
        self.explore_method = explore_method
        self.exploit_method = exploit_method
        self.al_feature = al_feature
        self.uncertain_type = uncertain_type
        # record the number of times each feature is observed by expert
        self.feature_observed_count = np.zeros(self.dataset.feature_num)
        self.feature_cov = np.mean(self.dataset.xs != 0, axis=0)

    def sample(self, stage="explore", al_model=None):
        is_candidate = np.repeat(True, len(self.dataset))
        if self.replace:
            is_candidate[np.array(self.sampled_idxs)] = False

        candidates = np.arange(len(self.dataset))[is_candidate]
        if len(candidates) == 0:
            # no remaining data for selection
            return -1
        else:
            if stage == "explore":
                if self.explore_method == "passive":
                    idx = self.rng.choice(candidates)
                elif self.explore_method == "count":
                    # explore based on observed count
                    feature_score = self.feature_cov / (self.feature_observed_count + 1)
                    xs_score = self.dataset.xs * feature_score
                    xs_score = np.sum(xs_score, axis=1)[candidates]
                    idx = self.rng.choice(candidates, p=xs_score/np.sum(xs_score))
                    self.feature_observed_count += self.dataset.xs[idx,:]
                elif self.explore_method == "count-exp":
                    feature_score = self.feature_cov / np.exp(- self.feature_observed_count)
                    xs_score = self.dataset.xs * feature_score
                    xs_score = np.sum(xs_score, axis=1)[candidates]
                    idx = self.rng.choice(candidates, p=xs_score / np.sum(xs_score))
                    self.feature_observed_count += self.dataset.xs[idx, :]

                else:
                    raise ValueError("Explore sample method not implemented.")

            else:
                if self.exploit_method == "uncertain":
                    assert al_model is not None
                    idx = uncertain_sample(self.dataset, candidates, al_model, self.al_feature,
                                           uncertain_type=self.uncertain_type)
                else:
                    raise ValueError("Exploit sample method not implemented.")

            self.sampled_idxs.append(idx)
            return idx