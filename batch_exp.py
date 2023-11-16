import sys
import os


datasets = [
    # "Youtube",
    "IMDB",
    # "Yelp",
    # "Amazon",
    # "BiasBios-professor-teacher",
    # "BiasBios-journalist-photographer",
]

lf_acc = 0.6
label_model = "snorkel"
al_model = "logistic"
filter_method = "Glasso"
sampler = "hybrid"
use_valid_labels = True
alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for dataset in datasets:
    for alpha in alpha_list:
        cmd = f"python icws.py --dataset {dataset} --label-model {label_model} " \
              f"--acc-threshold {lf_acc}  --sampler {sampler} --alpha {alpha}"
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
