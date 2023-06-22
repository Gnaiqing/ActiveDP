import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import optuna
# from utils import *
import pdb


def get_discriminator(model_type, prob_labels, input_dim, params=None, seed=None):
    if model_type == 'logistic':
        return LogReg(prob_labels, params, seed)
    elif model_type == 'logistic-torch':
        return LogRegTorch(prob_labels, params, seed)
    elif model_type == 'mlp-torch':
        return TorchMLP(h_sizes=[input_dim, 20, 20])
    else:
        raise ValueError('discriminator model not supported.')


class Classifier:
    """Classifier backbone
    """
    def tune_params(self, x_train, y_train, x_valid, y_valid, device=None):
        raise NotImplementedError

    def fit(self, xs, ys, device=None):
        raise NotImplementedError

    def predict(self, xs, device=None):
        raise NotImplementedError


class LogReg(Classifier):
    def __init__(self, prob_labels, params=None, seed=None):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'solver': ['liblinear'],
                'max_iter': [1000],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
            }
        self.params = params
        self.n_trials = 10
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed

    def tune_params(self, x_train, y_train, x_valid, y_valid, sample_weights=None, scoring='acc', device=None):
        search_space = self.params

        if self.prob_labels:
            x_train = np.vstack((x_train, x_train))
            weights = np.hstack((1.-y_train, y_train))
            y_train = np.hstack([-np.ones(len(y_train)), np.ones(len(y_train))])
            if sample_weights is not None:
                sample_weights = np.hstack((sample_weights, sample_weights)) * weights
            else:
                sample_weights = weights

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}

            model = LogisticRegression(**suggestions, random_state=self.seed)
            model.fit(x_train, y_train, sample_weights)            

            ys_pred = model.predict(x_valid)
            
            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                val_score = f1_score(y_valid, ys_pred)

            return val_score
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params


    def fit(self, xs, ys, sample_weights=None, device=None):
        if self.prob_labels:
            xs = np.vstack((xs, xs))
            weights = np.hstack((1.-ys, ys))
            ys = np.hstack([-np.ones(len(ys)), np.ones(len(ys))])
            if sample_weights is not None:
                sample_weights = np.hstack((sample_weights, sample_weights)) * weights
            else:
                sample_weights = weights

        if self.best_params is not None:
            model = LogisticRegression(**self.best_params, random_state=self.seed)
            model.fit(xs, ys, sample_weight=sample_weights)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')

    def predict(self, xs, device=None):
        return self.model.predict(xs)

    def predict_proba(self, xs, device=None):
        return self.model.predict_proba(xs)


class LogRegTorch(Classifier):
    def __init__(self, prob_labels, params=None, seed=None):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'lr': [1e-3, 1e-2],
                'l2': [1e-4, 1e-3, 1e-2],
                'n_epochs': [100],
                'patience': [3],
                'batch_size': [4096]
            }
        self.params = params
        self.n_trials = 10

        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed

    def tune_params(self, x_train, y_train, x_valid, y_valid, sample_weights=None, scoring='acc', device=None):
        seed = self.seed
        search_space = self.params
        train_dataset = LabeledDataset(x_train, y_train, weights=sample_weights)

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            batch_size = suggestions.pop('batch_size')

            torch.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            model = LogRegTorchBase(input_dim=x_train.shape[1], **suggestions)
            if device is not None:
                model = model.to(device)

            model.fit(train_loader, seed, device=device)
            ys_pred = model.predict_proba(torch.tensor(x_valid, dtype=torch.float), device=device)
            ys_pred = np.array([np.random.choice(np.where(y == np.max(y))[0]) for y in ys_pred])

            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                val_score = f1_score(y_valid, ys_pred)

            return val_score
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

    def fit(self, xs, ys, sample_weights=None, device=None):
        seed = self.seed
        dataset = LabeledDataset(xs, ys, sample_weights)
        batch_size = self.best_params.pop('batch_size')
        torch.manual_seed(seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LogRegTorchBase(input_dim=xs.shape[1], **self.best_params)
        if device is not None:
            model = model.to(device)

        model.fit(data_loader, seed=seed, device=device)
        self.model = model

    def predict(self, xs, device=None):
        ys_pred = self.predict_proba(xs, device=device)
        ys_pred = np.array([np.random.choice(np.where(y==np.max(y))[0]) for y in ys_pred])
        return ys_pred

    def predict_proba(self, xs, device=None):
        xs = torch.tensor(xs).float()
        if device is not None:
            xs = xs.to(device)
        ys_pred = self.model.predict_proba(xs)
        return ys_pred


class LogRegTorchBase(nn.Module):
    def __init__(self, input_dim, lr, l2, n_epochs, patience):
        super().__init__()
        self.linear_0 = nn.Linear(input_dim, 1)
        self.lr = lr
        self.l2 = l2
        self.n_epochs = n_epochs
        self.patience = patience

    def forward(self, x):
        logit = self.linear_0(x)
        return logit

    def fit(self, train_loader, seed=None, device=None):
        if seed is not None:
            torch.manual_seed(seed)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        
        pre_loss = np.inf
        patience = self.patience
        for epoch in range(self.n_epochs):
            running_loss = 0.
            for xs_tr, ys_tr, weights in train_loader:
                if device is not None:
                    xs_tr, ys_tr, weights = xs_tr.to(device), ys_tr.to(device), weights.to(device)

                optimizer.zero_grad()
                # loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                # loss = loss_fn(self(xs_tr), ys_tr.flatten())
                loss = F.binary_cross_entropy_with_logits(self(xs_tr), ys_tr, reduction='none')
                loss = (loss * weights).mean()
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            running_loss /= len(train_loader)

            if pre_loss - running_loss < 1e-4:
                patience -= 1
            else:
                patience = self.patience

            pre_loss = running_loss
            if patience < 0:
                break

    def predict_proba(self, xs, device=None):
        if device is not None:
            xs = xs.to(device)
        logits = self(xs)
        ys_pred = torch.sigmoid(logits).detach().cpu().numpy()
        ys_pred = np.hstack((1.-ys_pred, ys_pred))
        return ys_pred


# the following end model is adapted from IWS
class FeedforwardFlexible(torch.nn.Module):
    def __init__(self, h_sizes, activations):
        super(FeedforwardFlexible, self).__init__()

        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activations[k])

        self.layers.append(torch.nn.Linear(h_sizes[-1], 1))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def weight_reset(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


class TorchMLP(Classifier):
    def __init__(self, h_sizes=[150, 20, 20], activations=[torch.nn.ReLU(), torch.nn.ReLU()], optimizer='Adam',
                 optimparams={}, nepochs=200):
        self.model = FeedforwardFlexible(h_sizes, activations).float()
        self.optimizer = optimizer
        if optimizer == 'Adam':
            if optimparams:
                self.optimparams = optimparams
            else:
                self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}

        self.epochs = nepochs

    def tune_params(self, x_train, y_train, x_valid, y_valid, device=None):
        pass

    def fit(self, X, Y, batch_size=None, sample_weights=None, device=None):
        if device is not None:
            self.model = self.model.to(device)

        tinput = torch.from_numpy(X).to(torch.float)
        target = torch.from_numpy(Y.reshape(-1, 1))
        if device is not None:
            tinput = tinput.to(device)
            target = target.to(device)
        tweights = None
        if sample_weights is not None:
            tweights = torch.from_numpy(sample_weights.reshape(-1, 1)).to(torch.float)
            if device is not None:
                tweights = tweights.to(device)

        criterion = torch.nn.BCELoss(reduction='none')
        self.model.apply(weight_reset)

        trainX, trainy = tinput, target
        trainweight = None
        if tweights is not None:
            trainweight = tweights

        if self.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.model.parameters(),
                                          lr=1,
                                          max_iter=400,
                                          max_eval=15000,
                                          tolerance_grad=1e-07,
                                          tolerance_change=1e-04,
                                          history_size=10,
                                          line_search_fn=None)

            def closure():
                optimizer.zero_grad()
                mout = self.model(trainX)
                closs = criterion(mout, trainy)
                if tweights is not None:
                    closs = torch.mul(closs, trainweight).mean()
                else:
                    closs = closs.mean()

                closs.backward()
                return closs

            # only take one step (one epoch)
            optimizer.step(closure)
        else:
            optimizer = None
            if self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(), **self.optimparams)
            elif self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), **self.optimparams)
            lastloss = None
            tolcount = 0
            if batch_size is None:
                for nep in range(self.epochs):

                    out = self.model(trainX)
                    loss = criterion(out, trainy.to(torch.float))
                    if tweights is not None:
                        loss = torch.mul(loss, trainweight).mean()
                    else:
                        loss = loss.mean()

                    # early stopping
                    if lastloss is None:
                        lastloss = loss
                    else:
                        if lastloss - loss < 1e-04:
                            tolcount += 1
                        else:
                            tolcount = 0
                        if tolcount > 9:
                            break

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
            else:
                N = trainX.size()[0]
                dostop = False
                for nep in range(self.epochs):
                    permutation = torch.randperm(N)

                    for i in range(0, N, batch_size):
                        optimizer.zero_grad()
                        indices = permutation[i:i + batch_size]
                        batch_x, batch_y = trainX[indices], trainy[indices]

                        out = self.model(batch_x)
                        loss = criterion(out, batch_y)
                        if tweights is not None:
                            batch_weight = trainweight[indices]
                            loss = torch.mul(loss, batch_weight).mean()
                        else:
                            loss = loss.mean()

                        # early stopping
                        if lastloss is None:
                            lastloss = loss
                        else:
                            if lastloss - loss < 1e-04:
                                tolcount += 1
                            else:
                                tolcount = 0
                            if tolcount > 10:
                                dostop = True
                                break

                        loss.backward()

                        optimizer.step()
                    if dostop:
                        break

    def predict(self, xs, device=None):
        ys_pred = self.predict_proba(xs, device=device)
        ys_pred = np.array([np.random.choice(np.where(y == np.max(y))[0]) for y in ys_pred])
        return ys_pred

    def predict_proba(self, Xtest, device=None):
        with torch.no_grad():
            tXtest = torch.from_numpy(Xtest).to(torch.float)
            if device is not None:
                tXtest = tXtest.to(device)
                preds = self.model(tXtest).data.cpu().numpy()
            else:
                preds = self.model(tXtest).data.numpy()

            ys_pred = np.hstack((1. - preds, preds))
        return ys_pred


class LabeledDataset(Dataset):
    def __init__(self, xs, ys, weights=None):
        assert len(xs) == len(ys)
        self.xs = torch.tensor(xs).float()
        self.ys = torch.tensor(ys).float().view(-1, 1)
        if weights is not None:
            self.weights = torch.tensor(weights).float().view(-1, 1)
        else:
            self.weights = torch.ones_like(self.ys)
    
    def __getitem__(self, index):
        return (self.xs[index], self.ys[index], self.weights[index])

    def __len__(self):
        return len(self.xs)