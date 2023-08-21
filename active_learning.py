import sys
import os
import argparse
import pandas as pd
import numpy as np
import wandb
from data_utils import load_data
from sampler import get_sampler
from agent import SimulateAgent
from csd import StructureDiscovery
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from discriminator import get_discriminator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Learning Baseline')
    # paths
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="../ws_data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='Youtube')
    parser.add_argument('--dataset-sample-size', type=int, default=None)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--feature', type=str, default='tfidf')
    # dataset settings: for text dataset processing
    parser.add_argument("--stemmer", type=str, default=None)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--max_ngram", type=int, default=1)
    # framework settings
    parser.add_argument('--num-query', type=int, default=300)
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=10)
    # sampler settings
    parser.add_argument("--sampler", type=str, default="uncertain")
    parser.add_argument("--uncertain-type", type=str, default="entropy")
    # agent settings
    parser.add_argument("--agent", type=str, default="simulate")
    parser.add_argument("--label-error-rate", type=float, default=0.0)
    # model settings
    parser.add_argument("--end-model", type=str, default="logistic")
    # experiment settings
    parser.add_argument("--version", type=str, default="0.1")
    parser.add_argument("--display", action="store_true")
    parser.add_argument('--runs', type=int, nargs='+', default=range(5))

    args = parser.parse_args()
    save_dir = f'{args.root_dir}/{args.save_dir}'
    data_dir = f'{args.root_dir}/{args.data_dir}'

    group_id = wandb.util.generate_id()
    config_dict = vars(args)
    config_dict["group_id"] = group_id
    config_dict["method"] = "Active Learning"
    # load dataset
    train_dataset, valid_dataset, test_dataset = load_data(data_root=data_dir,
                                                           dataset_name=args.dataset,
                                                           valid_ratio=args.valid_ratio,
                                                           test_ratio=args.test_ratio,
                                                           sample_size=args.dataset_sample_size,
                                                           stemmer=args.stemmer,
                                                           max_ngram=args.max_ngram,
                                                           min_df=args.min_df,
                                                           max_df=args.max_df
                                                           )

    for run in args.runs:
        wandb.init(
            project="scalable-idp-baselines",
            config=config_dict
        )
        wandb.define_metric("test_acc", summary="mean")
        wandb.define_metric("test_auc", summary="mean")
        wandb.define_metric("test_f1", summary="mean")
        wandb.define_metric("train_precision", summary="mean")
        wandb.define_metric("train_coverage", summary="mean")
        seed_rng = np.random.default_rng(seed=run)
        seed = seed_rng.choice(10000)
        test_acc_list = []
        test_auc_list = []
        test_f1_list = []

        sampler = get_sampler(sampler_type=args.sampler,
                              dataset=train_dataset,
                              seed=seed)

        agent = SimulateAgent(train_dataset,
                              seed=seed,
                              label_error_rate=args.label_error_rate)
        end_model = None
        for t in range(args.num_query + 1):
            if t % args.train_iter == 0 and t > 0:
                print("Evaluating after {} iterations".format(t))
                labeled_dataset = sampler.create_labeled_dataset()
                tr_features = labeled_dataset.xs_feature
                y_tr = labeled_dataset.ys
                val_features = valid_dataset.xs_feature
                y_val = valid_dataset.ys

                if args.end_model == "logistic":
                    end_model = get_discriminator("logistic",
                                                  prob_labels=False,
                                                  input_dim=train_dataset.xs_feature.shape[1],
                                                  seed=seed)
                else:
                    raise ValueError(f"End model {args.end_model} not supported yet.")

                end_model.tune_params(tr_features, y_tr, val_features, y_val)
                end_model.fit(tr_features, y_tr)

                test_features = test_dataset.xs_feature
                y_test_predicted = end_model.predict(test_features)
                y_test_proba = end_model.predict_proba(test_features)
                test_acc = accuracy_score(test_dataset.ys, y_test_predicted)

                if test_dataset.n_class == 2:
                    test_auc = roc_auc_score(test_dataset.ys, y_test_proba[:, 1])
                    test_f1 = f1_score(test_dataset.ys, y_test_predicted)
                else:
                    test_auc = roc_auc_score(test_dataset.ys, y_test_proba, average="macro", multi_class="ovo")
                    test_f1 = f1_score(test_dataset.ys, y_test_predicted, average="macro")

                print("Test Accuracy: {}".format(test_acc))
                print("Test F1: {}".format(test_f1))
                print("Test AUC: {}".format(test_auc))
                test_acc_list.append(test_acc)
                test_auc_list.append(test_auc)
                test_f1_list.append(test_f1)
                coverage_tr = t / len(train_dataset)
                wandb.log(
                    {
                        "num_query": t,
                        "train_precision": 1.0,
                        "train_coverage": coverage_tr,
                        "test_acc": test_acc,
                        "test_auc": test_auc,
                        "test_f1": test_f1,
                    }
                )

            if args.sampler == "passive":
                idx = sampler.sample()
            elif args.sampler == "uncertain":
                idx = sampler.sample(al_model=end_model)

            if idx == -1:
                raise ValueError(f"No remaining data for sampling.")

            label, features = agent.query(idx)
            sampler.update_feedback(idx, label, features)
            # display query results
            print("Query: ", train_dataset.xs_text[idx])
            print("Label: ", train_dataset.label_names[label])

        avg_test_acc = np.mean(test_acc_list)
        avg_test_auc = np.mean(test_auc_list)
        avg_test_f1 = np.mean(test_f1_list)
        print("AVG Test Accuracy: {}".format(avg_test_acc))
        print("AVG Test F1: {}".format(avg_test_f1))
        print("AVG Test AUC: {}".format(avg_test_auc))
        wandb.finish()





