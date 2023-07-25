"""
Simulate the user who provide feedbacks based on observed instance
"""
import numpy as np
import abc
import os
from nltk.stem import PorterStemmer


class SentimentLexicon:
    def __init__(self, data_root, stemmer="porter"):
        pos_words_file = os.path.join(data_root, 'opinion_lexicon/positive-words.txt')
        neg_words_file = os.path.join(data_root, 'opinion_lexicon/negative-words.txt')

        pos_tokens = list()
        neg_tokens = list()
        with open(pos_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 35:
                    token = line.rstrip()
                    pos_tokens.append(token)
        with open(neg_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 35:
                    token = line.rstrip()
                    neg_tokens.append(token)

        if stemmer == "porter":
            self.stemmer = PorterStemmer()
            pos_tokens = [self.stemmer.stem(token) for token in pos_tokens]
            neg_tokens = [self.stemmer.stem(token) for token in neg_tokens]

        token_sentiment = {token: 1 for token in pos_tokens}
        token_sentiment.update({token: -1 for token in neg_tokens})

        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.token_sentiment = token_sentiment


    def tokens_to_sentiments(self, tokens):
        """Return sentiments of tokens in a sentence
        """
        sentiments = np.array([self.token_sentiment.get(token, 0) for token in tokens])

        return sentiments


    def tokens_with_sentiment(self, tokens, sentiment):
        """Return tokens with specified sentiment
        """
        sentiments = self.tokens_to_sentiments(tokens)
        tokens = np.array(tokens)[sentiments == sentiment]

        return tokens


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
    def __init__(self, dataset, seed, max_features=None, label_error_rate=0.0, lf_error_rate=0.0,
                 criterion="acc", acc_threshold=0.6, zero_feat=False, lexicon=None):
        super(SimulateAgent, self).__init__(dataset, seed)
        self.label_error_rate = label_error_rate  # error rate for providing a wrong label
        self.lf_error_rate = lf_error_rate  # error rate for providing a LF with accuracy below threshold
        self.criterion = criterion  # criterion for returning features
        self.acc_threshold = acc_threshold  # accuracy threshold for returning features
        self.max_features = max_features  # maximum number of features returned per instance
        self.zero_feat = zero_feat  # if set to true, can return feature with 0 value on selected instance
        self.selected_features = np.repeat(False, self.dataset.n_features())
        self.lexicon = lexicon

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
        if p_feat >= self.lf_error_rate:
            lf_accurate = True
        else:
            lf_accurate = False

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
                acc = np.mean(ys == label)  # note: the LF is designed based on user provided label.
                cov = np.mean(selected)

                if lf_accurate:
                    # the generated LF is better than random
                    if acc >= self.acc_threshold and not self.selected_features[j]:
                        candidate_features.append(j)
                        candidate_feature_names.append(self.dataset.feature_names[j])
                        candidate_accs.append(acc)
                        candidate_covs.append(cov)
                else:
                    # the generated LF is worse than random
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

            for j in selected_features:
                self.selected_features[j] = True

        elif self.criterion == "lexicon":
            # select lexicon with corresponding sentiment
            assert self.lexicon is not None
            sentiment = -1 if label == 0 else 1

            candidate_token_idxs = np.nonzero(xi != 0)[0]
            candidate_tokens = [self.dataset.feature_names[j] for j in candidate_token_idxs]
            tokens = self.lexicon.tokens_with_sentiment(candidate_tokens, sentiment)
            candidate_features = []
            for token in tokens:
                j = self.dataset.get_feature_idx(token)
                if j != -1:
                    selected = self.dataset.xs[:, j] == xi[j]
                    if not isinstance(selected, np.ndarray):
                        selected = selected.toarray().flatten()
                    ys = self.dataset.ys[selected]
                    acc = np.mean(ys == label)  # note: this is accuracy based on user provided label.
                    cov = np.mean(selected)
                    if acc >= self.acc_threshold and not self.selected_features[j]:
                        candidate_features.append(j)

            if self.max_features is not None and len(candidate_features) > self.max_features:
                selected_features = self.rng.choice(candidate_features, size=self.max_features, replace=False)
            else:
                selected_features = candidate_features

            for j in selected_features:
                self.selected_features[j] = True

        else:
            raise ValueError(f"Agent selection criterion {self.criterion} not supported.")

        return label, selected_features






