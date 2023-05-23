import os
import sys
import json
import gzip
import string
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import pdb
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import abc
import re


class AbstractDataset(abc.ABC):
    def __init__(self, xs, ys):
        assert len(xs) == len(ys)
        self.xs = xs
        self.ys = ys
        self.n_class = len(np.unique(self.ys))

    def n_features(self):
        return self.xs.shape[1]

    def __len__(self):
        return len(self.ys)

    @abc.abstractmethod
    def display(self):
        """
        Show dataset statistics and examples
        :return:
        """
        pass

    @abc.abstractmethod
    def to_dataframe(self):
        """
        Transform dataset to python dataframe
        :return:
        """
        pass

    @abc.abstractmethod
    def create_subset(self, indices, features, labels=None):
        """
        Create a sub with subset of instance and features
        :param indices: subset indices
        :param features: subset features
        :param labels: subset labels. If set, the original ys will be replaced
        :return:
        """
        pass

    @abc.abstractmethod
    def get_feature_idx(self, feature_name):
        pass

    @abc.abstractmethod
    def generate_label_matrix(self, lfs):
        """
        Generate label matrix based on label functions
        :param lfs:
        :return:
        """
        pass


class DiscreteDataset(AbstractDataset):
    def __init__(self, xs, ys, dataset_name, feature_names=None, label_names=None):
        """
        Dataset with discrete
        :param xs: input features with discrete variables
        :param ys: input labels
        """
        super(DiscreteDataset, self).__init__(xs, ys)
        self.dataset_name = dataset_name
        self.feature_names = feature_names
        self.vocabulary_ = {}
        if self.feature_names is not None:
            for i, f in enumerate(self.feature_names):
                self.vocabulary_[f] = i

        self.label_names = label_names

    def create_subset(self, indices, features, labels=None):
        """
        Create a sub with subset of instance and features
        :param indices: subset indices
        :param features: subset features
        :param labels: subset labels. If set, the original ys will be replaced
        :return:
        """
        sub_xs = self.xs[indices, features].toarray()
        if labels is None:
            sub_ys = self.ys[indices]
        else:
            sub_ys = labels

        if self.feature_names is not None:
            feature_names = self.feature_names[features]
        else:
            feature_names = None

        return DiscreteDataset(sub_xs, sub_ys, dataset_name=self.dataset_name,
                               feature_names=feature_names,
                               label_names=self.label_names)

    def to_dataframe(self):
        df = pd.DataFrame(data=self.xs, columns=self.feature_names)
        df.insert(len(df.columns), column="LABEL", value=self.ys)
        return df

    def get_feature_idx(self, feature_name):
        if feature_name in self.vocabulary_:
            return self.vocabulary_[feature_name]
        else:
            return -1

    def display(self):
        print(f"Displaying dataset {self.dataset_name}")
        print(f"Number of instances: {len(self.ys)}")
        print(f"Number of features: {self.xs.shape[1]}")
        print(f"Displaying head tuples:")
        df = pd.DataFrame(self.xs[:5, :], columns=self.feature_names)
        df.insert(len(df.columns), column="LABEL", value=self.ys[:5])
        print(df)

    def generate_label_matrix(self, lfs):
        L = []
        for (j, op, v, l) in lfs:
            wl = np.repeat(-1, len(self.ys))
            if op == "=":
                b = (self.xs[:, j] == v).view(-1)
            elif op == ">":
                b = (self.xs[:, j] > v).view(-1)
            elif op == "<":
                b = (self.xs[:, j] < v).view(-1)

            wl[b] = l
            L.append(wl)

        L = np.hstack(L)
        return L


# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

# preprocess text
def preprocessor(text):
    # lowercase the text
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = remove_punctuation(text)
    text = text.replace("\ufeff", "")
    # remove whitespaces and stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)
    return text

# Create a stemmer object
porterstemmer = PorterStemmer()

# Define a custom tokenizer function that applies stemming
def tokenize(text):
    tokens = text.split()
    # for token in tokens:
        # stemmed_token = porterstemmer.stem(token)
        # if not stemmed_token.isalpha():
        #     print("Stemmed token:")
        #     print(repr(stemmed_token))
        # else:
        #     stemmed_tokens.append(stemmed_token)
    stemmed_tokens = [porterstemmer.stem(token) for token in tokens]
    return stemmed_tokens


class TextDataset(AbstractDataset):
    def __init__(self, xs_text, ys, dataset_name, label_names=None, feature="tfidf", max_ngram=1, max_df=1.0, min_df=1,
                 max_features=None, stemmer="porter", count_vectorizer=None, pipeline=None):
        """
        Intialize text dataset
        :param xs_text: list of input texts or input feature array
        :param keyword_dict: dictionary containing all candidate keywords
        :param ys: labels
        :param feature: embeddings used to train end model
        :param count_vectorizer: transform text to array of binary features
        :param pipeline: transform text to array of embedding for end model
        """
        super(TextDataset, self).__init__(xs_text, ys)
        self.xs_text = xs_text
        self.ys = ys
        self.dataset_name = dataset_name
        self.label_names = label_names

        if count_vectorizer is None:
            if stemmer == "porter":
                self.count_vectorizer = CountVectorizer(preprocessor=preprocessor,
                                                        tokenizer=tokenize,
                                                        ngram_range=(1, max_ngram),
                                                        max_df=max_df,
                                                        min_df=min_df,
                                                        max_features=max_features,
                                                        binary=True)
            else:
                self.count_vectorizer = CountVectorizer(preprocessor=preprocessor,
                                                        ngram_range=(1, max_ngram),
                                                        max_df=max_df,
                                                        min_df=min_df,
                                                        max_features=max_features,
                                                        binary=True)
            self.xs = self.count_vectorizer.fit_transform(self.xs_text)
            self.feature_num = self.xs.shape[1]
            self.feature_names = self.count_vectorizer.get_feature_names_out().astype(str)
        else:
            self.count_vectorizer = count_vectorizer
            self.xs = self.count_vectorizer.transform(self.xs_text)
            self.feature_num = self.xs.shape[1]
            self.feature_names = self.count_vectorizer.get_feature_names_out().astype(str)

        # self.feature_dict = {}
        # if self.feature_names is not None:
        #     for i, f in enumerate(self.feature_names):
        #         self.feature_dict[f] = i

        if pipeline is None:
            if feature == 'tfidf':
                vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_df=0.9, max_features=1000)
                scalar = StandardScaler()
                self.xs_feature = vectorizer.fit_transform(xs_text).toarray()
                self.xs_feature = scalar.fit_transform(self.xs_feature)
                self.pipeline = (vectorizer, scalar)
                # self.pipeline = Pipeline([
                #     ('tfidf',
                #      TfidfVectorizer(strip_accents='ascii', stop_words='english', max_df=0.9, max_features=1000)),
                #     ('scaler', StandardScaler(with_mean=False))
                # ])
                # self.xs_feature = self.pipeline.fit_transform(xs_text)
            else:
                raise ValueError('Feature representation not supported.')
        else:
            # self.pipeline = pipeline
            # self.xs_feature = self.pipeline.transform(xs_text)
            (vectorizer, scalar) = pipeline
            self.xs_feature = vectorizer.transform(xs_text).toarray()
            self.xs_feature = scalar.transform(self.xs_feature)

    def get_feature_idx(self, feature_name):
        if feature_name in self.count_vectorizer.vocabulary_:
            return self.count_vectorizer.vocabulary_[feature_name]
        else:
            return -1

    def create_subset(self, indices, features, labels=None):
        """
        Create a sub with subset of instance and features
        :param indices: subset indices
        :param features: subset features
        :param labels: subset labels. If set, the original ys will be replaced
        :return:
        """
        sub_xs = self.xs[indices, :].toarray()

        filtered_features = []
        for j in features:
            if np.min(sub_xs[:, j]) != np.max(sub_xs[:, j]):
                filtered_features.append(j)

        features = np.array(filtered_features)
        sub_xs = sub_xs[:, features]

        if labels is None:
            sub_ys = self.ys[indices]
        else:
            sub_ys = labels


        if self.feature_names is not None:
            feature_names = self.feature_names[features]
        else:
            feature_names = None

        return DiscreteDataset(sub_xs, sub_ys, dataset_name=self.dataset_name,
                               feature_names=feature_names,
                               label_names=self.label_names)

    def to_dataframe(self):
        df = pd.DataFrame(data=self.xs.toarray(), columns=self.feature_names)
        df.insert(len(df.columns), column="LABEL", value=self.ys)
        return df

    def display(self, split="train"):

        print(f"Displaying dataset {self.dataset_name} [{split}]")
        print(f"Number of instances: {len(self.ys)}")
        print(f"Number of features: {self.xs.shape[1]}")
        print(f"Displaying head sentences: ")
        for i in range(5):
            print("Text:", self.xs_text[i])
            if self.label_names is not None:
                print("Label:", self.label_names[self.ys[i]])
            else:
                print("Label:", self.ys[i])

        print(f"Displaying head tuples:")
        df = pd.DataFrame(self.xs[:5, :].toarray(), columns=self.feature_names)
        df.insert(len(df.columns), column="LABEL", value=self.ys[:5])
        print(df)

    def generate_label_matrix(self, lfs):
        L = []
        for (j, op, v, l) in lfs:
            wl = np.repeat(-1, len(self.ys))
            if op == "=":
                b = (self.xs[:, j] == v).toarray().flatten()
            elif op == ">":
                b = (self.xs[:, j] > v).toarray().flatten()
            elif op == "<":
                b = (self.xs[:, j] < v).toarray().flatten()

            wl[b] = l
            L.append(wl.reshape(-1,1))

        L = np.hstack(L)
        return L


def tr_val_te_split(xs, ys, test_ratio, valid_ratio, rand_state):
    xs = np.array(xs)
    ys = np.array(ys)
    assert len(xs) == len(ys)
    N = len(xs)

    permuted_idxs = rand_state.permutation(N)
    num_test = int(N * test_ratio)
    num_valid = int(N * valid_ratio)

    train_idxs, test_idxs = permuted_idxs[:-num_test], permuted_idxs[-num_test:]
    train_idxs, valid_idxs = train_idxs[:-num_valid], train_idxs[-num_valid:]

    return (xs[train_idxs], ys[train_idxs], xs[valid_idxs], ys[valid_idxs],
            xs[test_idxs], ys[test_idxs], train_idxs, valid_idxs, test_idxs)


def load_data(data_root, dataset_name, valid_ratio, test_ratio, seed=0, sample_size=None,
              stemmer="porter", feature="tfidf", max_ngram=1, max_df=1.0, min_df=1, max_features=None):
    """
    Load dataset and split it
    :param data_root: dataset directory path
    :param dataset_name:
    :param valid_ratio:
    :param test_ratio:
    :param rand_state: for splitting dataset
    :param params: parameters for processing text data
    :return:
    """
    if dataset_name in ["Youtube", "IMDB", "Yelp", "Amazon", "Amazon-short", "Agnews"]:
        dataset_type = "text"
    elif dataset_name in []:
        dataset_type = "discrete"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if dataset_name == "Youtube":
        data_dir = os.path.join(data_root, "spam/data")
        files = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv',
                 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
        df_all = None
        for f in files:
            data_file = os.path.join(data_dir, f)
            df = pd.read_csv(data_file, sep=',', header=0, encoding='utf-8-sig')
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

        df = df_all.to_numpy()
        raw_texts = df[:, 3]
        labels = df[:, 4]
        labels = labels.astype(int)
        label_names = ["nonspam", "spam"]

    elif dataset_name == "IMDB":
        pos_dir = os.path.join(data_root, "aclImdb/train/pos")
        neg_dir = os.path.join(data_root, "aclImdb/train/neg")
        pos_files = os.listdir(pos_dir)
        pos_paths = [os.path.join(pos_dir, file) for file in pos_files]
        pos_texts = list()
        for path in pos_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                pos_texts.append(line)

        neg_files = os.listdir(neg_dir)
        neg_paths = [os.path.join(neg_dir, file) for file in neg_files]
        neg_texts = list()
        for path in neg_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                neg_texts.append(line)

        raw_texts = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)
        label_names = ["negative", "positive"]

    elif dataset_name == "Yelp":
        train_path = os.path.join(data_root, "yelp_review_polarity_csv/train.csv")
        test_path = os.path.join(data_root, "yelp_review_polarity_csv/test.csv")
        df_train = pd.read_csv(train_path, sep=',', header=None, encoding='utf-8-sig')
        df_test = pd.read_csv(test_path, sep=',', header=None, encoding='utf-8-sig')
        df_all = pd.concat([df_train, df_test])
        df = df_all.to_numpy()
        raw_texts = df[:, 1]
        labels = df[:, 0].astype(int) - 1
        label_names = ["negative", "positive"]

    elif dataset_name == "Amazon":
        train_path = os.path.join(data_root, "amazon_review_polarity_csv/train.csv")
        test_path = os.path.join(data_root, "amazon_review_polarity_csv/test.csv")
        df_train = pd.read_csv(train_path, sep=',', header=None, encoding='utf-8-sig')
        df_test = pd.read_csv(test_path, sep=',', header=None, encoding='utf-8-sig')
        df_all = pd.concat([df_train, df_test])
        df = df_all.to_numpy()
        raw_texts = df[:, 2]
        labels = df[:, 0].astype(int) - 1
        label_names = ["negative", "positive"]

    elif dataset_name == "Amazon-short":
        # short version of amazon review that only use title
        train_path = os.path.join(data_root, "amazon_review_polarity_csv/train.csv")
        test_path = os.path.join(data_root, "amazon_review_polarity_csv/test.csv")
        df_train = pd.read_csv(train_path, sep=',', header=None, encoding='utf-8-sig')
        df_test = pd.read_csv(test_path, sep=',', header=None, encoding='utf-8-sig')
        df_all = pd.concat([df_train, df_test])
        df = df_all.to_numpy()
        raw_texts = df[:, 1]
        labels = df[:, 0].astype(int) - 1
        label_names = ["negative", "positive"]

    elif dataset_name == "Agnews":
        train_path = os.path.join(data_root, "ag_news_csv/train.csv")
        test_path = os.path.join(data_root, "ag_news_csv/test.csv")
        df_train = pd.read_csv(train_path, sep=',', header=None, encoding='utf-8-sig')
        df_test = pd.read_csv(test_path, sep=',', header=None, encoding='utf-8-sig')
        df_all = pd.concat([df_train, df_test])
        df = df_all.to_numpy()
        raw_texts = df[:, 1]
        labels = df[:, 0].astype(int) - 1
        label_names = ["World", "Sports", "Business", "Sci/Tech"]

    else:
        # TODO: support discrete datasets
        xs = None
        feature_names = None
        label_names = None

    rand_state = np.random.default_rng(seed)
    if sample_size is not None and len(labels) > sample_size:
        # sample the dataset
        selected_indices = rand_state.choice(len(labels), size=sample_size, replace=False)
        labels = labels[selected_indices]
        if dataset_type == "text":
            raw_texts = raw_texts[selected_indices]
        else:
            xs = xs[selected_indices, :]

    if dataset_type == "text":
        train_xs_text, train_ys, valid_xs_text, valid_ys, test_xs_text, test_ys, train_idxs, valid_idxs, test_idxs = \
            tr_val_te_split(raw_texts, labels, test_ratio, valid_ratio, rand_state)
        trainset = TextDataset(train_xs_text, train_ys, dataset_name, feature=feature, max_ngram=max_ngram,
                               stemmer=stemmer,max_df=max_df,
                               min_df=min_df, max_features=max_features, label_names=label_names)
        validset = TextDataset(valid_xs_text, valid_ys, dataset_name, feature=feature, max_ngram=max_ngram,
                               stemmer=stemmer, max_df=max_df,
                               min_df=min_df, max_features=max_features,
                               count_vectorizer=trainset.count_vectorizer, pipeline=trainset.pipeline,
                               label_names=label_names)
        testset = TextDataset(test_xs_text, test_ys, dataset_name, feature=feature, max_ngram=max_ngram,
                              stemmer=stemmer, max_df=max_df,
                              min_df=min_df, max_features=max_features,
                              count_vectorizer=trainset.count_vectorizer, pipeline=trainset.pipeline,
                              label_names=label_names)

    elif dataset_type == "discrete":
        train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = \
            tr_val_te_split(xs, labels, test_ratio, valid_ratio, rand_state)
        trainset = DiscreteDataset(train_xs, train_ys, dataset_name, feature_names=feature_names,
                                   label_names=label_names)
        validset = DiscreteDataset(valid_xs, valid_ys, dataset_name, feature_names=feature_names,
                                   label_names=label_names)
        testset = DiscreteDataset(test_xs, test_ys, dataset_name, feature_names=feature_names, label_names=label_names)

    return trainset, validset, testset


def filter_abstain(L, ys):
    """
    Filter out abstain indices
    :param L: label matrix
    :param ys: labels
    :return:
    """
    non_abstain = (L!=-1).any(axis=1)
    L = L[non_abstain]
    ys = ys[non_abstain]
    return L, ys, non_abstain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../ws_data/")
    parser.add_argument("--dataset_name", type=str, default="youtube")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    # for text dataset processing
    parser.add_argument("--stemmer", type=str, default="porter")
    parser.add_argument("--min_df", type=int, default=20)
    parser.add_argument("--max_df", type=float, default=0.7)
    parser.add_argument("--max_ngram", type=int, default=1)

    args = parser.parse_args()
    train_dataset, valid_dataset, test_dataset = load_data(data_root=args.data_root,
                                                           dataset_name=args.dataset_name,
                                                           valid_ratio=args.valid_ratio,
                                                           test_ratio=args.test_ratio,
                                                           seed=args.seed,
                                                           stemmer=args.stemmer,
                                                           max_ngram=args.max_ngram,
                                                           min_df=args.min_df,
                                                           max_df=args.max_df)
    train_dataset.display(split="train")
    valid_dataset.display(split="valid")
    test_dataset.display(split="test")
