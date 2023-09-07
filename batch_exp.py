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
lf_filters = ["Glasso"]
al_models = ["logistic"]


for dataset in datasets:
    for lf_acc in lf_acc_thres:
        for lf_filter in lf_filters:
            for al_model in al_models:
                cmd = f"python main.py --dataset {dataset} --lf-acc {lf_acc} --sampler SEU --filter-method {lf_filter} --al-model {al_model}"
                if dataset in ["Amazon", "Yelp"]:
                    cmd += " --dataset-sample-size 25000"
                print(cmd)
                os.system(cmd)
