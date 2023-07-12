import sys
import os


datasets = [
    # "Youtube",
    "IMDB",
    "Yelp",
    "Amazon",
    # "BiasBios-professor-teacher",
    # "BiasBios-journalist-photographer",
]

lf_acc_thres = [0.6]
label_models = ["snorkel"]
al_models = ["logistic"]
filter_methods = ["Glasso"]
# filter_methods = ["Glasso", "ARD", "DT", "LinearSVRL2"]
samplers = ["uncertain"]
# samplers = ["passive", "uncertain", "QBC", "QUIRE", "LAL", "count", "SEU"]
use_valid_labels = True
criterion = "lexicon"

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lm in label_models:
            for sampler in samplers:
                for al_model in al_models:
                     for filter_method in filter_methods:
                        cmd = f"python icws.py --dataset {dataset} --label-model {lm} " \
                              f"--acc-threshold {lf_acc}  --sampler {sampler} --criterion {criterion}"
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
