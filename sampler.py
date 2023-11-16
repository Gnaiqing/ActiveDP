import numpy as np
from scipy.stats import entropy
from alipy import ToolBox
import abc


def get_sampler(sampler_type, dataset, seed, al_feature="tfidf", alpha=0.5):
    if sampler_type in ["uncertain", "QBC", "QUIRE", "EER", "density", "LAL"]:
        return ActiveSampler(dataset, al_feature, sampler_type, seed)
    elif sampler_type in ["passive", "SEU"]:
        return LFSampler(dataset, sampler_type, seed)
    elif sampler_type == "hybrid":
        return HybridSampler(dataset, seed, alpha=alpha)
    else:
        raise ValueError(f"Sampler {sampler_type} not supported.")


class Sampler(abc.ABC):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.sampled_idxs = list()
        self.sampled_labels = list()
        self.sampled_features = list()
        self.feature_mask = np.repeat(False, self.dataset.n_features())  # record whether a feature is selected.

    @abc.abstractmethod
    def sample(self, **kwargs):
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


class LFSampler(Sampler):
    """
    Sample based on LFs
    """
    def __init__(self, dataset, sample_method, seed):
        super(LFSampler, self).__init__(dataset, seed)
        self.sample_method = sample_method
        self.unlabeled_index = np.arange(len(self.dataset))
        self.rng = np.random.default_rng(seed)

    def sample(self, **kwargs):
        is_candidate = np.repeat(True, len(self.dataset))
        if len(self.sampled_idxs) > 0:
            is_candidate[np.array(self.sampled_idxs)] = False

        unlabeled_index = np.arange(len(self.dataset))[is_candidate]
        if len(unlabeled_index) == 0:
            return -1
        if len(unlabeled_index) > 1000:
            # subsample to reduce runtime
            unlabeled_index = self.rng.choice(unlabeled_index, size=1000, replace=False)
            unlabeled_index = np.sort(unlabeled_index)

        if self.sample_method == "passive":
            idx = self.rng.choice(unlabeled_index)
        elif self.sample_method == "SEU":
            # select by expected utility (Nemo)
            assert "y_probs" in kwargs
            if kwargs["y_probs"] is None:
                # fall back to passive sampling
                idx = self.rng.choice(unlabeled_index)
            else:
                uncertainty = entropy(kwargs["y_probs"], axis=1)[unlabeled_index]
                y_preds = np.argmax(kwargs["y_probs"], axis=1)[unlabeled_index]
                feature_mat = self.dataset.xs[unlabeled_index,:].toarray()
                wl_mat = []
                for i in range(self.dataset.n_class):
                    wl = - np.ones_like(feature_mat)
                    wl[feature_mat != 0] = i
                    wl_mat.append(wl)
                wl_mat = np.hstack(wl_mat)

                score = np.zeros_like(wl_mat)
                score[wl_mat != -1] = -1
                score[y_preds.reshape(-1, 1) == wl_mat] = 1  # record whether the weak labels are correct
                lf_score = np.sum(uncertainty.reshape(-1, 1) * score, axis=0)  # LF utility score
                lf_acc = np.sum(y_preds.reshape(-1, 1) == wl_mat, axis=0) / np.sum(wl_mat != -1, axis=0)
                lf_mask = wl_mat != -1  # whether a user may design LF based on x
                lf_probs = lf_mask * lf_acc + 1e-6
                lf_probs = lf_probs / np.sum(lf_probs, axis=1, keepdims=True)
                phi = np.sum(lf_score * lf_probs, axis=1)
                idx = unlabeled_index[np.argmax(phi)]
        else:
            raise ValueError(f"Sample method {self.sample_method} not supported.")

        self.sampled_idxs.append(idx)
        return idx


class ActiveSampler(Sampler):
    """
    Sample based on AL model
    """
    def __init__(self, dataset, al_feature, al_method, seed):
        super(ActiveSampler, self).__init__(dataset, seed)
        self.al_feature = al_feature
        self.al_method = al_method
        if self.al_feature == "tfidf":
            train_X = dataset.xs_feature
        else:
            train_X = dataset.xs

        train_y = dataset.ys
        self.alibox = ToolBox(X=train_X, y=train_y, query_type='AllLabels')
        self.train_index = self.alibox.IndexCollection(np.arange(len(dataset)))
        self.labeled_index = self.alibox.IndexCollection([0])
        self.labeled_index.difference_update([0])
        self.unlabeled_index = self.alibox.IndexCollection(np.arange(len(dataset)))

    def sample(self, al_model=None):
        if len(self.unlabeled_index.index) == 0:
            # no remaining data for selection
            return -1

        if al_model is None or self.al_method == "passive":
            strategy = self.alibox.get_query_strategy(strategy_name='QueryInstanceRandom')
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  batch_size=1)[0]
        elif self.al_method == "uncertain":
            strategy = self.alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  model=al_model,
                                  batch_size=1)[0]

        elif self.al_method == "QBC":
            strategy = self.alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  model=al_model,
                                  batch_size=1)[0]
        elif self.al_method == "EER":
            strategy = self.alibox.get_query_strategy(strategy_name='QueryExpectedErrorReduction')
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  model=al_model,
                                  batch_size=1)[0]

        elif self.al_method == "QUIRE":
            strategy = self.alibox.get_query_strategy(strategy_name='QueryInstanceQUIRE',
                                                      train_idx=self.train_index)
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  model=al_model,
                                  batch_size=1)[0]
        elif self.al_method == "density":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceGraphDensity",
                                                      train_idx=self.train_index)
            idx = strategy.select(label_index=self.labeled_index,
                                  unlabel_index=self.unlabeled_index,
                                  model=al_model,
                                  batch_size=1)[0]
        elif self.al_method == "LAL":
            lal = self.alibox.get_query_strategy(strategy_name="QueryInstanceLAL", cls_est=10, train_slt=False)
            lal.download_data()
            lal.train_selector_from_file(reg_est=30, reg_depth=5)
            idx = lal.select(label_index=self.labeled_index,
                             unlabel_index=self.unlabeled_index,
                             model=al_model,
                             batch_size=1)[0]

        else:
            raise ValueError(f"Acive Learning method {self.al_method} not supported.")

        self.labeled_index.update([idx])
        self.unlabeled_index.difference_update([idx])
        self.sampled_idxs.append(idx)
        return idx


class HybridSampler(Sampler):
    """
    Sample based on uncertainty from active learning model and label model.
    """
    def __init__(self, dataset, seed, alpha=0.5):
        super(HybridSampler, self).__init__(dataset, seed)
        self.rng = np.random.default_rng(seed)
        self.dataset = dataset
        self.alpha = alpha
        self.labeled_index = np.array([], dtype=int)
        self.unlabeled_index = np.arange(len(dataset))

    def sample(self, al_model=None, y_probs=None):
        if al_model is None:
            if y_probs is None:
                idx = self.rng.choice(self.unlabeled_index)
            else:
                uncertainty = entropy(y_probs, axis=1)[self.unlabeled_index]
                pos = np.argmax(uncertainty)
                idx = self.unlabeled_index[pos]
        else:
            al_probs = al_model.predict_proba(self.dataset.xs_feature)
            if y_probs is None:
                uncertainty = entropy(al_probs, axis=1)[self.unlabeled_index]
                pos = np.argmax(uncertainty)
                idx = self.unlabeled_index[pos]
            else:
                al_uncertainty = entropy(al_probs, axis=1)[self.unlabeled_index]
                dp_uncertainty = entropy(y_probs, axis=1)[self.unlabeled_index]
                uncertainty = np.power(al_uncertainty, self.alpha) * np.power(dp_uncertainty, 1-self.alpha)
                pos = np.argmax(uncertainty)
                idx = self.unlabeled_index[pos]

        self.labeled_index = np.union1d(self.labeled_index, idx)
        self.unlabeled_index = np.setdiff1d(self.unlabeled_index, idx)
        self.sampled_idxs.append(idx)
        return idx

