# for each run extract the average influence of the PF
import os
import pandas as pd
import numpy as np


fit_func = [
    "influence_seedSize", 
    "influence_time",
    "influence_communities",
    "influence_communities_time"
]

result = []
for directory in os.listdir(): 

    if 'exp1' in directory : 

        all_values = [directory]
        for ff in fit_func:
            mean_influence = []
            path = os.path.join(directory, ff)

            for file in os.listdir(path):
                if 'hv' not in file and 'time' not in file: 
                    df = pd.read_csv(os.path.join(path, file))
                    mean_influence.append(df['influence'].mean())

            all_values.append(round(np.mean(mean_influence),2))
        

        result.append(all_values)

result = pd.DataFrame(result)
result.columns = ['dataset', "influence_seedSize", "influence_time", "influence_communities", "influence_communities_time"]


result.to_csv('no_seed_comparison.csv')