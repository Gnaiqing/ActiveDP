import sys
import os


datasets = [
    # "IMDB",
    "Yelp",
    "Agnews",
    "Amazon",
    "Amazon-short",
    "BiasBios-professor-teacher",
    "BiasBios-professor-physician",
    "BiasBios-journalist-photographer",
    "BiasBios-painter-architect"
]

lf_acc_thres = [0.6]
label_models = ["mv", "snorkel"]
al_models = ["logistic"]
feature_noises = [0.00]

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lm in label_models:
            for al_model in al_models:
                for ferr in feature_noises:
                    cmd = f"python icws.py --dataset {dataset} --dataset-sample-size 3000 --label-model {lm} " \
                          f"--acc-threshold {lf_acc} --feature-error-rate {ferr}"
                    if al_model is not None:
                        cmd += f" --al-model {al_model}"
                    print(cmd)
                    os.system(cmd)
                    cmd = f"python icws.py --dataset {dataset} --dataset-sample-size 3000 --label-model {lm} --causal-filter " \
                          f"--csd-method Glasso --acc-threshold {lf_acc} --feature-error-rate {ferr}"
                    if al_model is not None:
                        cmd += f" --al-model {al_model}"
                    print(cmd)
                    os.system(cmd)
