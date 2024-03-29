"""
Scalable Interactive Data Programming
"""
import torch
import argparse
import numpy as np
import wandb
from data_utils import load_data, filter_abstain
from sampler import get_sampler
from agent import SimulateAgent, SentimentLexicon
from csd import StructureDiscovery
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from discriminator import get_discriminator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lf_utils import get_lf_stats, select_al_thresholds, merge_filtered_probs, check_filter, check_all_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scalable Interactive Data Programming')
    # paths
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="../ws_data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--dataset-sample-size', type=int, default=None)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--valid-sample-frac', type=float, default=None)
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
    parser.add_argument('--use-valid-labels', action="store_true")  # whether use validation set label for tuning
    parser.add_argument('--warmup-size', type=int, default=100)  # only used when use-valid-label set to false
    # sampler settings
    parser.add_argument("--sampler", type=str, default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5)  # only used when sampler==hybrid
    parser.add_argument("--uncertain-type", type=str, default="entropy")
    # agent settings
    parser.add_argument("--agent", type=str, default="simulate")
    parser.add_argument("--max-features", type=int, default=1)
    parser.add_argument("--criterion", type=str, default="acc", choices=["acc", "lexicon"])
    parser.add_argument("--acc-threshold", type=float, default=0.6)
    parser.add_argument("--label-error-rate", type=float, default=0.0)
    parser.add_argument("--lf-error-rate", type=float, default=0.0)
    # LF filter settings
    parser.add_argument("--filter-method", type=str, default="Glasso")
    # active learning settings
    parser.add_argument("--al-model", type=str, default="logistic")
    # model settings
    parser.add_argument("--label-model", type=str, default="snorkel")
    parser.add_argument("--use-soft-labels", action="store_true")
    parser.add_argument("--end-model", type=str, default="logistic")
    # experiment settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--display", action="store_true")
    parser.add_argument('--runs', type=int, nargs='+', default=range(5))

    args = parser.parse_args()
    save_dir = f'{args.root_dir}/{args.save_dir}'
    data_dir = args.data_dir

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
        args.device = 'cpu'

    group_id = wandb.util.generate_id()
    config_dict = vars(args)
    config_dict["group_id"] = group_id
    config_dict["method"] = "ActiveDP"
    # load dataset
    train_dataset, valid_dataset, test_dataset = load_data(data_root=data_dir,
                                                           dataset_name=args.dataset,
                                                           valid_ratio=args.valid_ratio,
                                                           test_ratio=args.test_ratio,
                                                           sample_size=args.dataset_sample_size,
                                                           stemmer=args.stemmer,
                                                           max_ngram=args.max_ngram,
                                                           min_df=args.min_df,
                                                           max_df=args.max_df,
                                                           valid_sample_frac=args.valid_sample_frac,
                                                           )

    train_dataset.display(split="train")

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
                              seed=seed,
                              alpha=args.alpha,
                              al_feature=args.feature)

        sentiment_lexicon = SentimentLexicon(data_dir)

        agent = SimulateAgent(train_dataset,
                              seed=seed,
                              max_features=args.max_features,
                              label_error_rate=args.label_error_rate,
                              lf_error_rate=args.lf_error_rate,
                              criterion=args.criterion,
                              acc_threshold=args.acc_threshold,
                              zero_feat=False,
                              lexicon=sentiment_lexicon)

        al_model = None
        label_model = None
        dp_coverage = 1.0
        al_coverage = 0.0

        for t in range(args.num_query + 1):
            if t % args.train_iter == 0 and t > 0:
                print("Evaluating after {} iterations".format(t))
                if args.filter_method is not None and (args.use_valid_labels or t > args.warmup_size):
                    # apply LF filtering
                    labeled_dataset = sampler.create_labeled_dataset(features="selected",
                                                                     drop_const_columns=True)
                    structure_learner = StructureDiscovery(labeled_dataset)
                    filtered_features = structure_learner.get_neighbor_nodes(method=args.filter_method,
                                                                             alpha=0.01,
                                                                             display=args.display)

                    filtered_feature_indices = []
                    for feature_name in filtered_features:
                        j = train_dataset.get_feature_idx(feature_name)
                        if j != -1:
                            filtered_feature_indices.append(j)

                    filtered_feature_indices = np.sort(filtered_feature_indices)
                    label_model, lfs = check_filter(
                        sampler=sampler,
                        label_model_type=args.label_model,
                        filtered_feature_indices=filtered_feature_indices,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        use_valid_labels=args.use_valid_labels,
                        seed=seed,
                        device=device
                    )
                    L_train = train_dataset.generate_label_matrix(lfs=lfs)
                    L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
                    lf_stats = get_lf_stats(L_train, train_dataset.ys)
                    L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
                    L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)

                else:
                    lfs = sampler.create_label_functions()
                    L_train = train_dataset.generate_label_matrix(lfs=lfs)
                    L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
                    lf_stats = get_lf_stats(L_train, train_dataset.ys)
                    L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
                    L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)

                    if not check_all_class(lfs, train_dataset.n_class):
                        label_model = None
                    else:
                        if args.label_model == "mv":
                            label_model = MajorityLabelVoter(cardinality=train_dataset.n_class)
                        elif args.label_model == "snorkel":
                            label_model = LabelModel(cardinality=train_dataset.n_class, device=device)
                            label_model.fit(L_train=L_tr_filtered, Y_dev=y_val_filtered, seed=seed)
                        else:
                            raise ValueError(f"Label model {args.label_model} not supported yet.")

                if label_model is None:
                    wandb.log(
                        {
                            "num_query": t,
                            "train_precision": np.nan,
                            "train_coverage": np.nan,
                            "dp_coverage": np.nan,
                            "al_coverage": np.nan,
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
                    y_tr_predicted = label_model.predict(L_tr_filtered, tie_break_policy="random")

                    if args.al_model is not None and (args.use_valid_labels or t > args.warmup_size):
                        # combine AL model with PWS model
                        if args.al_model == "logistic":
                            al_model = LogisticRegression(random_state=seed)
                        elif args.al_model == "decision-tree":
                            al_model = DecisionTreeClassifier(random_state=seed)
                        else:
                            raise ValueError(f"AL model {args.al_model} not supported.")

                        labeled_dataset = sampler.create_labeled_dataset(features="all", drop_const_columns=False)
                        al_model.fit(labeled_dataset.xs_feature, labeled_dataset.ys)
                        tr_features = train_dataset.xs_feature
                        val_features = valid_dataset.xs_feature
                        if args.use_valid_labels:
                            val_labels = valid_dataset.ys
                            al_val_probs = al_model.predict_proba(val_features)
                            dp_val_preds = label_model.predict(L_val_filtered)
                            theta = select_al_thresholds(al_val_probs, dp_val_preds, val_filtered_indices, val_labels)
                        else:
                            # split the labeled subset for training and validation purpose
                            L_labeled = labeled_dataset.generate_label_matrix(lfs=lfs)
                            x_train, x_sp, y_train, y_sp, wl_train, wl_sp = train_test_split(
                                labeled_dataset.xs, labeled_dataset.ys, L_labeled, test_size=0.5)

                            if args.al_model == "logistic":
                                al_model_sp = LogisticRegression(random_state=seed)
                            elif args.al_model == "decision-tree":
                                al_model_sp = DecisionTreeClassifier(random_state=seed)

                            al_model_sp.fit(x_train, y_train)
                            al_sp_probs = al_model_sp.predict_proba(x_sp)
                            dp_sp_probs = label_model.predict(wl_sp)
                            dp_indices = np.arange(len(wl_sp))
                            theta = select_al_thresholds(al_sp_probs, dp_sp_probs, dp_indices, y_sp)

                        # track the respective coverage of DP and AL
                        al_tr_probs = al_model.predict_proba(tr_features)
                        tr_filtered_indices_dp = tr_filtered_indices  # indices labeled by DP
                        al_conf = np.max(al_tr_probs, axis=1)
                        tr_filtered_indices_al = al_conf > theta
                        dp_coverage = np.mean(tr_filtered_indices_dp & (~tr_filtered_indices_al))
                        al_coverage = np.mean(tr_filtered_indices_al)

                        y_tr_predicted_soft, y_tr_predicted, tr_filtered_indices = merge_filtered_probs(
                            al_tr_probs, y_tr_predicted_soft, y_tr_predicted, tr_filtered_indices, theta)
                        y_tr_filtered = train_dataset.ys[tr_filtered_indices]
                    else:
                        dp_coverage = len(y_tr_filtered) / len(train_dataset)
                        al_coverage = 0.0

                    precision_tr = accuracy_score(y_tr_filtered, y_tr_predicted)
                    coverage_tr = len(y_tr_filtered) / len(train_dataset)
                    print('Recovery Precision: {}'.format(precision_tr))
                    print('Coverage: {}'.format(coverage_tr))

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
                        test_auc = roc_auc_score(test_dataset.ys, y_test_proba[:,1])
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
                            "dp_coverage": dp_coverage,
                            "al_coverage": al_coverage,
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

            if label_model is not None:
                y_probs = label_model.predict_proba(L_train)
            else:
                y_probs = None

            if args.sampler == "hybrid":
                idx = sampler.sample(al_model=al_model, y_probs=y_probs)
            elif args.sampler == "SEU":
                idx = sampler.sample(y_probs=y_probs)
            else:
                idx = sampler.sample(al_model=al_model)

            # query the agent
            label, features = agent.query(idx)
            sampler.update_feedback(idx, label, features)

            # update label model and AL model
            lfs = sampler.create_label_functions()
            if not check_all_class(lfs, train_dataset.n_class):
                label_model = None
            else:
                L_train = train_dataset.generate_label_matrix(lfs=lfs)
                L_valid = valid_dataset.generate_label_matrix(lfs=lfs)
                lf_stats = get_lf_stats(L_train, train_dataset.ys)
                L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)
                L_val_filtered, y_val_filtered, val_filtered_indices = filter_abstain(L_valid, valid_dataset.ys)

                if args.label_model == "mv" or len(lfs) < 3 or np.min(y_val_filtered) == np.max(y_val_filtered):
                    label_model = MajorityLabelVoter(cardinality=train_dataset.n_class)
                elif args.label_model == "snorkel":
                    label_model = LabelModel(cardinality=train_dataset.n_class, device=device)
                    try:
                        label_model.fit(L_train=L_tr_filtered, Y_dev=y_val_filtered, seed=seed)
                    except RuntimeError:
                        label_model = MajorityLabelVoter(cardinality=train_dataset.n_class)
                else:
                    raise ValueError(f"Label model {args.label_model} not supported yet.")

            if args.al_model == "logistic":
                al_model = LogisticRegression(random_state=seed)
            elif args.al_model == "decision-tree":
                al_model = DecisionTreeClassifier(random_state=seed)
            elif args.al_model is None:
                al_model = None
            else:
                raise ValueError(f"AL model {args.al_model} not supported.")

            if t < 10 or len(np.unique(sampler.sampled_labels)) < train_dataset.n_class:
                al_model = None
            elif al_model is not None:
                labeled_dataset = sampler.create_labeled_dataset(features="all", drop_const_columns=False)
                al_model.fit(labeled_dataset.xs_feature, labeled_dataset.ys)

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
