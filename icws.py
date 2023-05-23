"""
Interactive Causal Weak Supervision
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import wandb
from data_utils import load_data, filter_abstain
from sampler import Sampler
from agent import SimulateAgent
from csd import CausalDiscovery
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from discriminator import get_discriminator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive Causal Weak Supervision')
    # paths
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--feature', type=str, default='tfidf')
    # dataset settings: for text dataset processing
    parser.add_argument("--stemmer", type=str, default="porter")
    parser.add_argument("--min_df", type=int, default=20)
    parser.add_argument("--max_df", type=float, default=0.7)
    parser.add_argument("--max_ngram", type=int, default=1)
    # framework settings
    parser.add_argument('--num-query', type=int, default=300)
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=10)
    parser.add_argument('--warmup-size', type=int, default=100)
    # sampler settings
    parser.add_argument("--sampler", type=str, default="passive")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bs-size", type=int, default=5)
    # agent settings
    parser.add_argument("--agent", type=str, default="simulate")
    parser.add_argument("--max-features", type=int, default=1)
    parser.add_argument("--criterion", type=str, default="acc")
    parser.add_argument("--acc-threshold", type=float, default=0.6)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--zero-feat", action="store_true")
    # causal structure discovery settings
    parser.add_argument("--causal-filter", action="store_true")
    parser.add_argument("--csd-method", type=str, default=None)
    parser.add_argument("--ci-test", type=str, default="discrete")
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    # model settings
    parser.add_argument("--label-model", type=str, default="mv")
    parser.add_argument("--use-soft-labels", action="store_true")
    parser.add_argument("--end-model", type=str, default="logistic")
    # experiment settings
    parser.add_argument("--display", action="store_true")
    parser.add_argument('--runs', type=int, nargs='+', default=range(5))

    args = parser.parse_args()
    save_dir = f'{args.root_dir}/{args.save_dir}'
    data_dir = f'{args.root_dir}/{args.data_dir}'

    group_id = wandb.util.generate_id()

    # load dataset
    train_dataset, valid_dataset, test_dataset = load_data(data_root=data_dir,
                                                           dataset_name=args.dataset,
                                                           valid_ratio=args.valid_ratio,
                                                           test_ratio=args.test_ratio,
                                                           stemmer=args.stemmer,
                                                           max_ngram=args.max_ngram,
                                                           min_df=args.min_df,
                                                           max_df=args.max_df
                                                           )

    train_dataset.display(split="train")


    for run in args.runs:
        wandb.init(
            project="icws",
            config={
                "dataset": args.dataset,
                "sampler": args.sampler,
                "agent": args.agent,
                "acc-threshold": args.acc_threshold,
                "causal-filter": args.causal_filter,
                "causal-warmup": args.warmup_size,
                "csd-method": args.csd_method,
                "bootstrap": args.bootstrap,
                "label-model": args.label_model,
                "end-model": args.end_model,
                "group-id": group_id
            }
        )

        test_acc_list = []
        test_auc_list = []
        test_f1_list = []
        sampler = Sampler(dataset=train_dataset, seed=run)
        agent = SimulateAgent(train_dataset,
                              run,
                              max_features=args.max_features,
                              error_rate=args.error_rate,
                              criterion=args.criterion,
                              acc_threshold=args.acc_threshold,
                              zero_feat=args.zero_feat)

        for t in range(args.num_query + 1):
            if t % args.train_iter == 0 and t > 0:
                print("Evaluating after {} iterations".format(t))
                if args.causal_filter and t >= args.warmup_size:
                    if args.bootstrap:
                        causal_features = []
                        for _ in range(args.bs_size):
                            bs_dataset = sampler.create_bootstrap_dataset()
                            causal_learner = CausalDiscovery(bs_dataset)
                            bs_causal_features = causal_learner.get_neighbor_nodes(method=args.csd_method,
                                                                                   alpha=args.ci_alpha,
                                                                                   display=args.display)
                            causal_features = causal_features + bs_causal_features

                        causal_features = np.unique(causal_features)
                    else:
                        labeled_dataset = sampler.create_labeled_dataset()
                        causal_learner = CausalDiscovery(labeled_dataset)
                        causal_features = causal_learner.get_neighbor_nodes(method=args.csd_method,
                                                                            alpha=args.ci_alpha,
                                                                            display=args.display)

                    print("Causal features:", causal_features)
                    causal_feature_indices = []
                    for feature_name in causal_features:
                        j = train_dataset.get_feature_idx(feature_name)
                        if j != -1:
                            causal_feature_indices.append(j)

                    causal_feature_indices = np.sort(causal_feature_indices)
                    lfs = sampler.create_label_functions(causal_feature_indices)
                else:
                    lfs = sampler.create_label_functions()

                class_exist = np.repeat(False, train_dataset.n_class)
                for (j, op, v, l) in lfs:
                    class_exist[l] = True

                if not np.all(class_exist):
                    print("Some class do not have LF. Skip training label model.")
                    wandb.log(
                        {
                            "num_query": t,
                            "train_precision": np.nan,
                            "train_coverage": np.nan,
                            "test_acc": np.nan,
                            "test_auc": np.nan,
                            "test_f1": np.nan,
                        }
                    )
                else:
                    L_train = train_dataset.generate_label_matrix(lfs=lfs)
                    L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
                    if args.label_model == "mv":
                        label_model = MajorityLabelVoter()
                    elif args.label_model == "snorkel":
                        label_model = LabelModel()
                    else:
                        raise ValueError(f"Label model {args.label_model} not supported yet.")

                    L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
                    L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)
                    if args.label_model == "snorkel":
                        label_model.fit(L_train=L_tr_filtered, Y_dev=y_val_filtered)

                    y_tr_predicted_soft = label_model.predict_proba(L_tr_filtered)
                    y_tr_predicted = label_model.predict(L_tr_filtered, tie_break_policy="random")

                    precision_tr = accuracy_score(y_tr_filtered, y_tr_predicted)
                    coverage_tr = len(y_tr_filtered) / len(train_dataset)
                    print('Recovery Precision: {}'.format(precision_tr))
                    print('Coverage: {}'.format(coverage_tr))

                    if args.end_model == "logistic":
                        end_model = get_discriminator("logistic", args.use_soft_labels)
                        # disc_model.tune_params(xs_tr, ys_tr, valid_dataset.xs_feature, valid_dataset.ys, sample_weights)
                        # disc_model.fit(xs_tr, ys_tr, sample_weights)
                    else:
                        raise ValueError(f"End model {args.end_model} not supported yet.")

                    tr_features = train_dataset.xs_feature[tr_filtered_indices,:]
                    val_features = valid_dataset.xs_feature
                    y_val = valid_dataset.ys
                    if args.use_soft_labels:
                        end_model.tune_params(tr_features, y_tr_predicted_soft, val_features, y_val)
                        end_model.fit(tr_features, y_tr_predicted_soft)
                    else:
                        end_model.tune_params(tr_features, y_tr_predicted, val_features, y_val)
                        end_model.fit(tr_features, y_tr_predicted)

                    test_features = test_dataset.xs_feature
                    y_test_predicted = end_model.predict(test_features)
                    y_test_proba = end_model.predict_proba(test_features)
                    test_acc = accuracy_score(test_dataset.ys, y_test_predicted)
                    test_f1 = f1_score(test_dataset.ys, y_test_predicted)
                    if test_dataset.n_class == 2:
                        test_auc = roc_auc_score(test_dataset.ys, y_test_proba[:,1])
                    else:
                        test_auc = roc_auc_score(test_dataset.ys, y_test_proba)

                    print("Test Accuracy: {}".format(test_acc))
                    print("Test F1: {}".format(test_f1))
                    print("Test AUC: {}".format(test_auc))
                    test_acc_list.append(test_acc)
                    test_auc_list.append(test_auc)
                    test_f1_list.append(test_f1)
                    wandb.log(
                        {
                            "num_query": t,
                            "train_precision": precision_tr,
                            "train_coverage": coverage_tr,
                            "test_acc": test_acc,
                            "test_auc": test_auc,
                            "test_f1": test_f1,
                        }
                    )

            if args.sampler == "passive":
                constraints = []
            else:
                raise ValueError(f"Sampler {args.sampler} not implemented.")

            idx = sampler.sample(constraints)
            if idx == -1:
                raise ValueError(f"No valid data for sampling.")

            # query the agent
            label, features = agent.query(idx)
            sampler.update_feedback(idx, label, features)

            # display query results
            print("Query: ", train_dataset.xs_text[idx])
            print("Label: ", train_dataset.label_names[label])
            selected_features = [train_dataset.feature_names[j] for j in features]
            print("Features: ", selected_features)

        avg_test_acc = np.mean(test_acc_list)
        avg_test_auc = np.mean(test_auc_list)
        avg_test_f1 = np.mean(test_f1_list)
        print("AVG Test Accuracy: {}".format(avg_test_acc))
        print("AVG Test F1: {}".format(avg_test_f1))
        print("AVG Test AUC: {}".format(avg_test_auc))
        wandb.finish()









