import sys
import os

datasets = ["IMDB"]

# datasets = ["IMDB",
#             "Yelp",
#             "Agnews",
#             "Amazon",
#             "Amazon-short",
#             "BiasBios-professor-teacher",
#             "BiasBios-professor-physician",
#             "BiasBios-journalist-photographer",
#             "BiasBios-painter-architect"]

lf_acc_thres = [0.6]
label_models = ["mv", "snorkel"]
feature_noises = [0.05, 0.10, 0.15]

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lm in label_models:
            for ferr in feature_noises:
                cmd = f"python icws.py --dataset {dataset} --dataset-sample-size 3000 --label-model {lm} " \
                      f"--acc-threshold {lf_acc} --feature-error-rate {ferr}"
                print(cmd)
                os.system(cmd)
                cmd = f"python icws.py --dataset {dataset} --dataset-sample-size 3000 --label-model {lm} --causal-filter " \
                      f"--csd-method Glasso --acc-threshold {lf_acc} --feature-error-rate {ferr}"
                print(cmd)
                os.system(cmd)
