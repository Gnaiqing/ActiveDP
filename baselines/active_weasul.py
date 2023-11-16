import sys
import os
import argparse
import pandas as pd
import numpy as np
import wandb
from data_utils import load_data, filter_abstain
from sampler import get_sampler
from agent import SimulateAgent
from discriminator import get_discriminator
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lf_utils import get_lf_stats, check_all_class
from aw_label_model import AWLabelModel
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active WeaSuL')
    # paths
    parser.add_argument('--root_dir', type=str, default='../')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="../ws_data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='Youtube')
    parser.add_argument('--dataset-sample-size', type=int, default=None)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--feature', type=str, default='tfidf')
    # dataset settings: for text dataset processing
    parser.add_argument("--stemmer", type=str, default="porter")
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=0.7)
    parser.add_argument("--max_ngram", type=int, default=1)
    # framework settings
    parser.add_argument('--num-query', type=int, default=300)
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=10)
    # sampler settings
    parser.add_argument("--sampler", type=str, default="hybrid")
    # agent settings
    parser.add_argument("--agent", type=str, default="simulate")
    parser.add_argument("--label-error-rate", type=float, default=0.0)
    parser.add_argument("--acc-threshold", type=float, default=0.6)
    # model settings
    parser.add_argument("--use-soft-labels", action="store_true")
    parser.add_argument("--end-model", type=str, default="logistic")
    # experiment settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--display", action="store_true")
    parser.add_argument('--runs', type=int, nargs='+', default=range(5))

    args = parser.parse_args()
    save_dir = f'{args.root_dir}/{args.save_dir}'
    data_dir = f'{args.root_dir}/{args.data_dir}'

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
        args.device = 'cpu'

    group_id = wandb.util.generate_id()
    config_dict = vars(args)
    config_dict["group_id"] = group_id
    config_dict["method"] = "Active WeaSuL"
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
                              max_features=1,
                              label_error_rate=args.label_error_rate,
                              criterion="acc",
                              acc_threshold=args.acc_threshold)

        label_model = None
        end_model = None
        for t in range(args.num_query + 1):
            if t % args.train_iter == 0 and t > 0:
                print("Evaluating after {} iterations".format(t))
                if label_model is None:
                    wandb.log(
                        {
                            "num_query": t,
                            "train_precision": np.nan,
                            "train_coverage": np.nan,
                            "test_acc": np.nan,
                            "test_auc": np.nan,
                            "test_f1": np.nan,
                            "lf_acc_avg": lf_stats["lf_acc_avg"],
                            "lf_acc_std": lf_stats["lf_acc_std"],
                            "lf_cov_avg": lf_stats["lf_cov_avg"],
                            "lf_cov_std": lf_stats["lf_cov_std"],
                            "lf_num": lf_stats["lf_num"]
                        }
                    )
                else:
                    y_tr_predicted_soft = label_model.predict_proba(L_tr_filtered)
                    y_tr_predicted = label_model.predict(L_tr_filtered)
                    precision_tr = accuracy_score(y_tr_filtered, y_tr_predicted)
                    coverage_tr = len(y_tr_filtered) / len(train_dataset)
                    print('Recovery Precision: {}'.format(precision_tr))
                    print('Coverage: {}'.format(coverage_tr))

                    if np.min(y_tr_predicted) != np.max(y_tr_predicted):
                        end_model = get_discriminator(args.end_model, args.use_soft_labels,
                                                      input_dim=train_dataset.xs_feature.shape[1],
                                                      seed=seed)

                        tr_features = train_dataset.xs_feature[tr_filtered_indices, :]
                        val_features = valid_dataset.xs_feature
                        y_val = valid_dataset.ys
                        if args.use_soft_labels:
                            end_model.tune_params(tr_features, y_tr_predicted_soft, val_features, y_val, device=device)
                            end_model.fit(tr_features, y_tr_predicted_soft, device=device)
                        else:
                            end_model.tune_params(tr_features, y_tr_predicted, val_features, y_val, device=device)
                            end_model.fit(tr_features, y_tr_predicted, device=device)

                        test_features = test_dataset.xs_feature
                        y_test_predicted = end_model.predict(test_features, device=device)
                        y_test_proba = end_model.predict_proba(test_features, device=device)
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
                        wandb.log(
                            {
                                "num_query": t,
                                "train_precision": precision_tr,
                                "train_coverage": coverage_tr,
                                "test_acc": test_acc,
                                "test_auc": test_auc,
                                "test_f1": test_f1,
                                "lf_acc_avg": lf_stats["lf_acc_avg"],
                                "lf_acc_std": lf_stats["lf_acc_std"],
                                "lf_cov_avg": lf_stats["lf_cov_avg"],
                                "lf_cov_std": lf_stats["lf_cov_std"],
                                "lf_num": lf_stats["lf_num"]
                            }
                        )
                    else:
                        print("Only one class label exist. Skip training end model")
                        test_acc, test_auc, test_f1 = np.nan, np.nan, np.nan
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
                                "lf_acc_avg": lf_stats["lf_acc_avg"],
                                "lf_acc_std": lf_stats["lf_acc_std"],
                                "lf_cov_avg": lf_stats["lf_cov_avg"],
                                "lf_cov_std": lf_stats["lf_cov_std"],
                                "lf_num": lf_stats["lf_num"]
                            }
                        )

            lfs = sampler.create_label_functions()
            if len(lfs) > 0:
                L_train = train_dataset.generate_label_matrix(lfs=lfs)
                L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
                lf_stats = get_lf_stats(L_train, train_dataset.ys)
                L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
                L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)

                if check_all_class(lfs, train_dataset.n_class):
                    label_model = AWLabelModel(n_class=train_dataset.n_class)
                    ground_truth_labels = - np.ones(len(train_dataset))
                    ground_truth_labels[np.array(sampler.sampled_idxs)] = sampler.sampled_labels
                    ground_truth_labels = ground_truth_labels[tr_filtered_indices]
                    label_model.fit(L_train=L_tr_filtered, L_valid=L_val_filtered, y_valid=y_val_filtered,
                                    ground_truth_labels=ground_truth_labels)

            if label_model is not None:
                y_probs = label_model.predict_proba(L_train)
            else:
                y_probs = None

            idx = sampler.sample(y_probs=y_probs)
            # query the agent
            label, features = agent.query(idx)
            sampler.update_feedback(idx, label, features)

        avg_test_acc = np.mean(test_acc_list)
        avg_test_auc = np.mean(test_auc_list)
        avg_test_f1 = np.mean(test_f1_list)
        print("AVG Test Accuracy: {}".format(avg_test_acc))
        print("AVG Test F1: {}".format(avg_test_f1))
        print("AVG Test AUC: {}".format(avg_test_auc))
        wandb.finish()





