import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from celf_pareto_fronts import find_best
import re


def aggregate_pfs(directory): 

    df_combined = pd.DataFrame()
    for f in os.listdir(directory): 
        pattern = re.compile("run-([0-9]+).csv")
        if bool(pattern.match(f)): 
            df = pd.read_csv(os.path.join(directory, f))
            df_combined = pd.concat([df_combined, df], axis=0, ignore_index=True)

    return df_combined

def boxplot(directory): 
    label = directory.replace('exp1_out_', '')
    print(label)

    df_final = pd.DataFrame()

    for f in ['influence_communities_time', 'influence_communities', 'influence_time']: 
        try: 
            df = aggregate_pfs(os.path.join(directory, f))
            df['fitness_function '] = [f] * len(df)
            df_final = pd.concat([df_final, df], axis = 0, ignore_index=True)
        except FileNotFoundError as f: 
            print(f)

    if not df_final.empty: 
        sns.set(rc={'figure.figsize':(9, 5)})
        sns.boxplot(data = df_final, x='fitness_function ', y='influence')
        plt.title(label)
        # plt.savefig('result_comparison/' + label + '/' + 'pf_noSeed_boxplot.png')
        plt.show()


if "__main__" == __name__: 
    for directory in os.listdir(): 
        if 'exp1_out' in directory: 
            print(directory)
            boxplot(directory)
