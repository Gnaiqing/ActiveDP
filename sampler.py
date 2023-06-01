import numpy as np
import pandas as pd
from data_utils import AbstractDataset, TextDataset, DiscreteDataset
import abc


class Sampler:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.sampled_idxs = list()
        self.candidate_history = list()  # include pairs of (candidate_list_size, mask)
        self.sampled_labels = list()
        self.sampled_features = list()
        self.feature_mask = np.repeat(False, self.dataset.n_features())

    def sample(self, constraints: list, replace=True):
        """
        Randomly sample a data point that satisfy schema. Return idx and record all candidates
        :param constraints: constraint list on feature values
        :param replace: whether sample with replacement
        :return:
        """
        is_candidate = np.repeat(True, len(self.dataset))
        # if not replace:  # TODO: fix it and consider the smir
        #     is_candidate[np.array(self.sampled_idxs)] = False

        for (j, op, v) in constraints:
            if op == "=":
                mask = self.dataset.xs[:,j].A1 == v
            elif op == ">":
                mask = self.dataset.xs[:,j].A1 > v
            elif op == "<":
                mask = self.dataset.xs[:,j].A1 < v
            else:
                raise ValueError(f"Opterator {op} not supported.")

            is_candidate = is_candidate & mask

        candidates = np.arange(len(self.dataset))[is_candidate]
        if len(candidates) == 0:
            # no valid data following constraints. Return -1
            return -1
        else:
            self.candidate_history.append((len(candidates), is_candidate))
            idx = self.rng.choice(candidates)
            self.sampled_idxs.append(idx)
            return idx

    def update_feedback(self, idx, label, features=None):
        assert idx == self.sampled_idxs[-1]
        self.sampled_labels.append(label)
        if features is not None:
            self.sampled_features.append(features)
            for j in features:
                self.feature_mask[j] = True

    def create_label_functions(self, causal_features=None):
        lfs = []
        for idx, label, features in zip(self.sampled_idxs, self.sampled_labels, self.sampled_features):
            if causal_features is not None:
                features = [f for f in features if f in causal_features]

            for f in features:
                v = self.dataset.xs[idx, f]
                lf = (f, "=", v, label)  # (feature_idx, op, value, label)
                if lf not in lfs:
                    lfs.append(lf)

        return lfs

    def create_labeled_dataset(self, features="selected", drop_const_columns=True):
        """
        Create labeled dataset using expert annotation
        :return:
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

    def create_bootstrap_dataset(self, dataset_size=None, features="selected", strategy="msir", drop_const_columns=True):
        """
        Create bootstrap dataset using labeled subset
        :param dataset_size: size for bootstrap dataset
        :param features: "selected", or "all"
        :param strategy: "msir" for multiple sampling importance resampling. "passive" for random sampling
        :param drop_const_columns: whether drop columns with constant value when creating dataset
        :return:
        """
        if strategy == "msir":
            weights = np.zeros(len(self.sampled_idxs))
            for i in range(len(self.sampled_idxs)):
                idx = self.sampled_idxs[i]
                for (l, mask) in self.candidate_history:
                    if mask[idx]:
                        weights[i] += 1 / l

            weights = 1 / weights
            weights = weights / np.sum(weights)
        elif strategy == "passive":
            weights = np.ones(len(self.sampled_idxs))
            weights = weights / np.sum(weights)
        else:
            raise ValueError(f"Bootstrap stragegy {strategy} not implemented.")

        if dataset_size is None:
            dataset_size = len(self.sampled_idxs)

        list_idxs = self.rng.choice(len(self.sampled_idxs), dataset_size, p=weights)
        subset_idxs = np.array(self.sampled_idxs)[list_idxs]
        subset_labels = np.array(self.sampled_labels)[list_idxs]
        if features == "selected":
            features = np.arange(self.dataset.n_features())[self.feature_mask]
        elif features == "all":
            features = np.arange(self.dataset.n_features())

        bs_dataset = self.dataset.create_subset(subset_idxs, features, labels=subset_labels, drop_const_columns=drop_const_columns)
        return bs_dataset








