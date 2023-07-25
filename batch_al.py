import sys
import os


datasets = [
    "Youtube",
    "IMDB",
    "Yelp",
    "Amazon",
    "BiasBios-professor-teacher",
    "BiasBios-journalist-photographer",
]

lf_acc_thres = [0.6]
label_err_rates = [0.05, 0.10, 0.15]

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for label_err in label_err_rates:
            cmd = f"python active_learning.py --dataset {dataset} --label-error-rate {label_err}"
            if dataset in ["Yelp", "Amazon"]:
                cmd += " --dataset-sample-size 25000"

            print(cmd)
            os.system(cmd)
