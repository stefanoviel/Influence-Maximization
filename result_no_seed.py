# for each run extract the average influence of the PF
import os
import pandas as pd
import numpy as np


fit_func = [
    "influence_time",
    "influence_communities",
    "influence_communities_time"
]

directory = "exp1_out_facebook_combined_4-IC"


for ff in fit_func:
    mean_influence = []
    path = os.path.join(directory, ff)

    for file in os.listdir(path):
        if 'hv' not in file and 'time' not in file: 
            df = pd.read_csv(os.path.join(path, file))
            mean_influence.append(df['influence'].mean())

    print(np.mean(mean_influence))