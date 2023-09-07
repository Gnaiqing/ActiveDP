import torch
import argparse
import numpy as np
import wandb
# from data_utils_activeDP import load_data
from data_utils import filter_abstain, load_data
from sampler import get_sampler
from agent import SimulateAgent, SentimentLexicon
from label_model import get_label_model
from discriminator import get_discriminator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lf_utils import get_lf_stats, check_all_class
from utils import select_al_thresholds, merge_filtered_probs, check_filter, get_filtered_indices, get_al_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ActiveDP')
    # paths
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="../ws_data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='Youtube')
    parser.add_argument('--dataset-sample-size', type=int, default=None)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--valid-sample-frac', type=float, default=None)
    parser.add_argument('--feature', type=str, default='tfidf')
    parser.add_argument("--stemmer", type=str, default='porter')
    # framework settings
    parser.add_argument('--num-query', type=int, default=50)
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=5)
    parser.add_argument('--use-valid-labels', type=bool, default=True)  # whether use validation set label for tuning
    # sampler settings
    parser.add_argument("--sampler", type=str, default="QBC")
    parser.add_argument("--lf-sample-method", type=str, default="passive")  # only used when sampler==hybrid
    parser.add_argument("--al-sample-method", type=str, default="QBC")  # only used when sampler==hybrid
    # agent settings
    parser.add_argument("--lf-simulate", type=str, default="acc", choices=["acc", "lexicon"])
    parser.add_argument("--lf-acc", type=float, default=0.6)
    parser.add_argument("--label-error-rate", type=float, default=0.0)
    parser.add_argument("--lf-error-rate", type=float, default=0.0)
    # LF filter settings
    parser.add_argument("--filter-method", type=str, default="Glasso")
    parser.add_argument("--ci-alpha", type=float, default=0.01)
    # active learning settings
    parser.add_argument("--al-model", type=str, default="logistic")
    # model settings
    parser.add_argument("--label-model", type=str, default="snorkel")
    parser.add_argument("--soft-training", type=bool, default=True)
    parser.add_argument("--end-model", type=str, default="logistic")
    # experiment settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablate-lm-tuning", action="store_true")
    parser.add_argument("--ablate-em-tuning", action="store_true")
    parser.add_argument('--runs', type=int, nargs='+', default=range(5))
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    if args.filter_method == "None":
        args.filter_method = None
    if args.al_model == "None":
        args.al_model = None

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
                                                           max_ngram=1,
                                                           min_df=1,
                                                           max_df=0.9,
                                                           valid_sample_frac=args.valid_sample_frac,
                                                           )
    # train_dataset, valid_dataset, test_dataset = load_data(data_root=data_dir,
    #                                                        dataset_name=args.dataset,
    #                                                        feature=args.feature,
    #                                                        test_ratio=args.test_ratio,
    #                                                        valid_ratio=args.valid_ratio,
    #                                                        warmup_ratio=0.0,
    #                                                        rand_state=np.random.RandomState(0),
    #                                                        )

    train_dataset.display(split="train")
    warmup_size = 100
    seed_rng = np.random.default_rng(seed=args.seed)
    for run in args.runs:
        wandb.init(
            project="ActiveDP",
            config=config_dict
        )
        wandb.define_metric("test_acc", summary="mean")
        wandb.define_metric("test_auc", summary="mean")
        wandb.define_metric("test_f1", summary="mean")
        wandb.define_metric("train_precision", summary="mean")
        wandb.define_metric("train_coverage", summary="mean")
        seed = seed_rng.choice(10000)

        test_acc_list = []
        test_auc_list = []
        test_f1_list = []

        sampler = get_sampler(sampler_type=args.sampler,
                              dataset=train_dataset,
                              seed=seed,
                              lf_sample_method=args.lf_sample_method,
                              al_sample_method=args.al_sample_method)

        sentiment_lexicon = SentimentLexicon(data_dir)

        agent = SimulateAgent(train_dataset,
                              seed=seed,
                              max_features=1,
                              label_error_rate=args.label_error_rate,
                              lf_error_rate=args.lf_error_rate,
                              criterion=args.lf_simulate,
                              acc_threshold=args.lf_acc,
                              lexicon=sentiment_lexicon)

        al_model = None
        label_model = None
        dp_coverage = 1.0
        al_coverage = 0.0

        for t in range(args.num_query + 1):
            if t % args.train_iter == 0 and t > 0:
                print("Evaluating after {} iterations".format(t))
                if len(np.unique(sampler.sampled_labels)) != train_dataset.n_class:
                    # some class have not been sampled
                    label_model = None
                    lfs = sampler.create_label_functions()
                    L_train = train_dataset.generate_label_matrix(lfs=lfs)
                    lf_stats = get_lf_stats(L_train, train_dataset.ys)

                elif args.filter_method is not None and (args.use_valid_labels or t > warmup_size):
                    # apply LF filtering
                    try:
                        filtered_feature_indices = get_filtered_indices(sampler, args.filter_method, args.ci_alpha)
                    except FloatingPointError:
                        filtered_feature_indices = np.nonzero(sampler.feature_mask)[0]
                    # check whether filtering improves label quality
                    label_model, lfs = check_filter(
                        sampler=sampler,
                        label_model_type=args.label_model,
                        filtered_feature_indices=filtered_feature_indices,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        use_valid_labels=args.use_valid_labels,
                        seed=seed,
                        device=device,
                        tune_params=not args.ablate_lm_tuning,
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
                        label_model = get_label_model(args.label_model, cardinality=train_dataset.n_class)
                        if args.label_model == "snorkel":
                            # label_model.fit(L_tr=L_tr_filtered, L_val=L_val_filtered, ys_val=y_val_filtered,
                            #                 tune_params=not args.ablate_lm_tuning)
                            label_model.fit(L_tr=L_tr_filtered, L_val=L_valid, ys_val=valid_dataset.ys,
                                            tune_params=not args.ablate_lm_tuning)

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
                    y_tr_predicted = label_model.predict(L_tr_filtered)

                    if args.al_model is not None and (args.use_valid_labels or t > warmup_size):
                        # combine AL model with PWS model
                        if args.al_model == "logistic":
                            al_model = LogisticRegression(random_state=seed)
                        elif args.al_model == "decision-tree":
                            al_model = DecisionTreeClassifier(random_state=seed)
                        else:
                            raise ValueError(f"AL model {args.al_model} not supported.")

                        labeled_dataset = sampler.create_labeled_dataset()
                        al_model.fit(labeled_dataset.xs_feature, labeled_dataset.ys)

                        if args.use_valid_labels:
                            val_labels = valid_dataset.ys
                            al_val_probs = al_model.predict_proba(valid_dataset.xs_feature)
                            dp_val_preds = label_model.predict(L_val_filtered)
                            theta = select_al_thresholds(al_val_probs, dp_val_preds, val_filtered_indices, val_labels)
                        else:
                            # split the labeled subset for training and validation purpose
                            L_labeled = labeled_dataset.generate_label_matrix(lfs=lfs)
                            x_train, x_sp, y_train, y_sp, wl_train, wl_sp = train_test_split(
                                labeled_dataset.xs, labeled_dataset.ys, L_labeled, test_size=0.5)

                            al_model_sp = get_al_model(args.al_model, seed)
                            al_model_sp.fit(x_train, y_train)
                            al_sp_probs = al_model_sp.predict_proba(x_sp)
                            dp_sp_probs = label_model.predict(wl_sp)
                            dp_indices = np.arange(len(wl_sp))
                            theta = select_al_thresholds(al_sp_probs, dp_sp_probs, dp_indices, y_sp)

                        # track the respective coverage of DP and AL
                        al_tr_probs = al_model.predict_proba(train_dataset.xs_feature)
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

                    end_model = get_discriminator(args.end_model, args.soft_training,
                                                  input_dim=train_dataset.xs_feature.shape[1],
                                                  seed=seed)

                    tr_features = train_dataset.xs_feature[tr_filtered_indices, :]
                    val_features = valid_dataset.xs_feature
                    y_val = valid_dataset.ys
                    if args.soft_training:
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
                al_sample_prob = al_coverage / (al_coverage + dp_coverage)
                idx = sampler.sample(al_sample_prob=al_sample_prob, al_model=al_model, y_probs=y_probs)
            elif args.sampler == "SEU":
                idx = sampler.sample(y_probs=y_probs)
            else:
                idx = sampler.sample(al_model=al_model)
            if idx == -1:
                raise ValueError(f"No remaining data for sampling.")

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
                L_tr_filtered, y_tr_filtered, tr_filtered_indices = filter_abstain(L_train, train_dataset.ys)

                if len(lfs) < 3 or args.label_model == "mv":
                    label_model = get_label_model("mv", cardinality=train_dataset.n_class)
                else:
                    label_model = get_label_model(args.label_model, cardinality=train_dataset.n_class)
                    label_model.fit(L_tr_filtered, L_valid, valid_dataset.ys, tune_params=not args.ablate_lm_tuning)

            if len(np.unique(sampler.sampled_labels)) == train_dataset.n_class:
                al_model = get_al_model(args.al_model, seed)
            else:
                al_model = None

            if al_model is not None:
                labeled_dataset = sampler.create_labeled_dataset()
                al_model.fit(labeled_dataset.xs_feature, labeled_dataset.ys)

            # display query results
            print("Query: ", train_dataset.xs_text[idx])
            print("Label: ", label)
            selected_features = [train_dataset.feature_names[j] for j in features]
            print("Features: ", selected_features)

        avg_test_acc = np.mean(test_acc_list)
        avg_test_auc = np.mean(test_auc_list)
        avg_test_f1 = np.mean(test_f1_list)
        print("AVG Test Accuracy: {}".format(avg_test_acc))
        print("AVG Test F1: {}".format(avg_test_f1))
        print("AVG Test AUC: {}".format(avg_test_auc))
        wandb.finish()
