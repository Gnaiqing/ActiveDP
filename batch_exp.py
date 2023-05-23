import sys
import os

datasets = ["youtube"]
csd_methods = ["PC", "GES", "GIES", "ARD", "Glasso"]
alpha_choices = {
    "PC": [0.2, 0.3, 0.4],
    "Glasso": [0.01, 0.03, 0.05]
}