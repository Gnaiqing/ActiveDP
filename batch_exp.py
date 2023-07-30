import sys
import os


datasets = [
    "Youtube",
    # "IMDB",
    # "Yelp",
    # "Amazon",
    # "BiasBios-professor-teacher",
    # "BiasBios-journalist-photographer",
]

lf_acc_thres = [0.6]
label_models = ["snorkel"]
al_models = ["logistic"]
filter_methods = ["Glasso"]
samplers = ["hybrid"]
use_valid_labels = True
valid_sample_fracs = [0.25, 0.5, 0.75]

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lm in label_models:
            for sampler in samplers:
                for al_model in al_models:
                    for filter_method in filter_methods:
                        for valid_sample_frac in valid_sample_fracs:
                            cmd = f"python icws.py --dataset {dataset} --label-model {lm} " \
                                  f"--acc-threshold {lf_acc}  --sampler {sampler} " \
                                  f"--valid-sample-frac {valid_sample_frac} --runs 0 1 2"
                            if dataset in ["Yelp", "Amazon"]:
                                cmd += " --dataset-sample-size 25000"
                            if al_model is not None:
                                cmd += f" --al-model {al_model}"
                            if filter_method is not None:
                                cmd += f" --filter-method {filter_method}"
                            if use_valid_labels:
                                cmd += " --use-valid-labels"

                            print(cmd)
                            os.system(cmd)
