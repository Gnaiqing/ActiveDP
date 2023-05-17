"""
Simulate the user who provide feedbacks based on observed instance
"""
import numpy as np
import abc


class AbstractAgent(abc.ABC):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng()

    @abc.abstractmethod
    def query(self, idx):
        """
        query the label and indicative features based on idx
        :param idx:
        :return:
        """
        pass


class SimulateAgent(AbstractAgent):
    def __init__(self, dataset, seed, max_features=None, error_rate=0.0, criterion="acc", acc_threshold=0.6, zero_feat=False):
        super(SimulateAgent, self).__init__(dataset, seed)
        self.error_rate = error_rate  # error rate for labelling
        self.criterion = criterion  # criterion for returning features
        self.acc_threshold = acc_threshold  # accuracy threshold for returning features
        self.max_features = max_features  # maximum number of features returned per instance
        self.zero_feat = zero_feat  # if set to true, can return feature with 0 value on selected instance
        self.selected_features = np.repeat(False, self.dataset.n_features())

    def query(self, idx):
        xi = self.dataset.xs[idx,:]
        if not isinstance(xi, np.ndarray):
            xi = xi.toarray()

        xi = xi.flatten()
        yi = self.dataset.ys[idx]

        p = self.rng.random()
        if p >= self.error_rate:
            label = yi
        else:
            candidates = [c for c in range(self.dataset.n_class) if c != yi]
            label = self.rng.choice(candidates)

        candidate_features = []
        candidate_feature_names = []
        if self.criterion == "acc":
            candidate_accs = []
            candidate_covs = []
            for j in range(self.dataset.n_features()):
                if (not self.zero_feat) and (xi[j] == 0):
                    continue
                selected = self.dataset.xs[:,j] == xi[j]
                if not isinstance(selected, np.ndarray):
                    selected = selected.toarray().flatten()
                ys = self.dataset.ys[selected]
                acc = np.mean(ys == yi)
                cov = np.mean(selected)
                # accuracy above threshold and feature not selected before
                if acc > self.acc_threshold and not self.selected_features[j]:
                    candidate_features.append(j)
                    candidate_feature_names.append(self.dataset.feature_names[j])
                    candidate_accs.append(acc)
                    candidate_covs.append(cov)

            if self.max_features is not None and len(candidate_features) > self.max_features:
                p = np.array(candidate_covs) / np.sum(candidate_covs)
                selected_features = self.rng.choice(candidate_features, size=self.max_features, replace=False, p=p)
            else:
                selected_features = candidate_features

            selected_feature_names = [self.dataset.feature_names[j] for j in selected_features]

            for j in selected_features:
                self.selected_features[j] = True

        else:
            raise ValueError(f"Agent selection criterion {self.criterion} not supported.")

        return label, selected_features






