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

def name_abreviation(name): 
    if name == 'influence_communities_time': 
        return 'I.C.T.'
    elif name == 'influence_communities': 
        return 'I.C.'
    elif name == 'influence_time': 
        return 'I.T.'

def boxplot(directory): 
    label = directory.replace('exp1_out_', '')
    print(label)

    df_final = pd.DataFrame()

    for f in ['influence_communities_time', 'influence_communities', 'influence_time']: 
        try: 
            df = aggregate_pfs(os.path.join(directory, f))
            df['fitness_function '] = [name_abreviation(f)] * len(df)
            df_final = pd.concat([df_final, df], axis = 0, ignore_index=True)
        except FileNotFoundError as f: 
            print(f)
    return df_final

        


if "__main__" == __name__: 
    dfs = []
    labels = []
    axs = []
    for directory in os.listdir(): 
        if 'exp1_out' in directory: 
            print(directory)
            labels.append(directory)
            dfs.append(boxplot(directory))

    plt.figure(figsize=(12,9))

    axs.append(plt.subplot(2,2,1))
    axs.append(plt.subplot(2,2,2))
    axs.append(plt.subplot(2,2,3))
    axs.append(plt.subplot(2,2,4))

    dfs_labels = [(x, l) for x, l in zip(dfs, labels) if not x.empty]

    
    for n, (data, label) in enumerate(dfs_labels): 

        # print(dfs_final)
        sns.boxplot(data = data, x='fitness_function ', y="influence", ax = axs[n]).set(ylabel='% influence')
        axs[n].set_title(label)

    # plt.savefig('result_comparison/' + label + '/' + 'box_plot_' + label + '.jpg')
    plt.subplots_adjust(hspace=0.25)
    plt.show()