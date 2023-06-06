import sys
import os


datasets = [
    # "IMDB",
    # "Yelp",
    # "Amazon",
    # "BiasBios-professor-teacher",
    "BiasBios-professor-physician",
    "BiasBios-journalist-photographer",
    "BiasBios-painter-architect"
]

lf_acc_thres = [0.6]
label_models = ["snorkel"]
al_models = [None, "logistic"]
filter_methods = [None, "Glasso"]
use_valid_labels = True

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lm in label_models:
            for al_model in al_models:
                 for filter_method in filter_methods:
                    cmd = f"python icws.py --dataset {dataset} --dataset-sample-size 3000 --label-model {lm} " \
                          f"--acc-threshold {lf_acc} "
                    if al_model is not None:
                        cmd += f" --al-model {al_model}"
                    if filter_method is not None:
                        cmd += f" --filter-method {filter_method}"
                    if use_valid_labels:
                        cmd += " --use-valid-labels"
                    print(cmd)
                    os.system(cmd)
