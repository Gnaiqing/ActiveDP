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

baseline_files = [
    "active_weasul.py",
    "nashaat.py"
]


for dataset in datasets:
    for baseline in baseline_files:
        cmd = f"python {baseline} --dataset {dataset}"
        if dataset in ["Yelp", "Amazon"]:
            cmd += " --dataset-sample-size 25000"

        print(cmd)
        os.system(cmd)
