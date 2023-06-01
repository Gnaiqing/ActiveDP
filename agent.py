"""
Simulate the user who provide feedbacks based on observed instance
"""
import numpy as np
import abc


class AbstractAgent(abc.ABC):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def query(self, idx):
        """
        query the label and indicative features based on idx
        :param idx:
        :return:
        """
        pass


class SimulateAgent(AbstractAgent):
    def __init__(self, dataset, seed, max_features=None, label_error_rate=0.0, feature_error_rate=0.0,
                 criterion="acc", acc_threshold=0.6, zero_feat=False):
        super(SimulateAgent, self).__init__(dataset, seed)
        self.label_error_rate = label_error_rate  # error rate for providing a wrong label
        self.feature_error_rate = feature_error_rate  # error rate for giving a feature not indicative of provided label
        self.criterion = criterion  # criterion for returning features
        self.acc_threshold = acc_threshold  # accuracy threshold for returning features
        self.max_features = max_features  # maximum number of features returned per instance
        self.zero_feat = zero_feat  # if set to true, can return feature with 0 value on selected instance
        self.selected_features = np.repeat(False, self.dataset.n_features())

    def query(self, idx):
        xi = self.dataset.xs[idx,:]
        text = self.dataset.xs_text[idx]
        if not isinstance(xi, np.ndarray):
            xi = xi.toarray()

        xi = xi.flatten()
        yi = self.dataset.ys[idx]

        p = self.rng.random()
        if p >= self.label_error_rate:
            label = yi
        else:
            candidates = [c for c in range(self.dataset.n_class) if c != yi]
            label = self.rng.choice(candidates)

        p_feat = self.rng.random()
        if p_feat >= self.feature_error_rate:
            is_feature_accurate = True
        else:
            is_feature_accurate = False

        candidate_features = []
        candidate_feature_names = []
        if self.criterion == "acc":
            candidate_accs = []
            candidate_covs = []
            for j in range(self.dataset.n_features()):
                if (not self.zero_feat) and (xi[j] == 0):
                    continue

                checking_feature = self.dataset.feature_names[j]
                selected = self.dataset.xs[:,j] == xi[j]
                if not isinstance(selected, np.ndarray):
                    selected = selected.toarray().flatten()
                ys = self.dataset.ys[selected]
                acc = np.mean(ys == label)  # note: this is accuracy based on user provided label.
                cov = np.mean(selected)

                if is_feature_accurate:
                    # accuracy above threshold and feature not selected before
                    if acc >= self.acc_threshold and not self.selected_features[j]:
                        candidate_features.append(j)
                        candidate_feature_names.append(self.dataset.feature_names[j])
                        candidate_accs.append(acc)
                        candidate_covs.append(cov)
                else:
                    # select low accuracy features
                    if acc < self.acc_threshold and not self.selected_features[j]:
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






