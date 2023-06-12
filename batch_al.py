import sys
import os


datasets = [
    "Youtube",
    "IMDB",
    "Yelp",
    "Amazon",
    "BiasBios-professor-teacher",
    "BiasBios-professor-physician",
    "BiasBios-journalist-photographer",
    "BiasBios-painter-architect"
]

lf_acc_thres = [0.6]

for dataset in datasets:
    for lf_acc in lf_acc_thres:
        cmd = f"python active_learning.py --dataset {dataset}"
        if dataset in ["Yelp", "Amazon"]:
            cmd += " --dataset-sample-size 25000"

        print(cmd)
        os.system(cmd)
