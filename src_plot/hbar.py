import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_initials(fitness_name): 
    if fitness_name == 'influence_seedSize_time': 
        return 'I.S.T.'
    elif fitness_name == 'influence_seedSize_communities': 
        return 'I.S.C.'
    elif fitness_name == 'influence_seedSize':
        return 'I.S.'
    elif fitness_name == 'influence_seedSize_communities_time':
        return 'I.S.C.T'


for directory in os.listdir('result_comparison'): 
    df_combined = pd.DataFrame()

    for file in os.listdir(os.path.join('result_comparison', directory, 'hvs')): 
        print(os.path.join('result_comparison', directory, 'hvs', file))
        df = pd.read_csv(os.path.join('result_comparison', directory, 'hvs', file))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['fitness_function'] = get_initials(file.replace('_hvs.csv', ''))
        df_combined = pd.concat([df_combined, df], axis=0)
    
    df_combined = df_combined[['hv_influence_seed', 'hv_influence_seedSize_time', 'hv_influence_seedSize_communities', 'fitness_function']]
    df_new = []
    for index, row in df_combined.iterrows():
        df_new.append([row["fitness_function"], "hv_influence_seedSize" , row["hv_influence_seed"], ])
        df_new.append([row["fitness_function"], "hv_influence_seedSize_time", row["hv_influence_seedSize_time"], ])
        df_new.append([row["fitness_function"], "hv_influence_seedSize_communities", row["hv_influence_seedSize_communities"], ])

    df_new = pd.DataFrame(df_new)
    df_new.columns = ['fitness_function', 'HV dimensions', 'HV values']
    g = sns.catplot(data=df_new, x='HV dimensions',y ='HV values', kind="bar",hue="fitness_function", legend = False,  height=4.5, aspect=2.2)
    g.fig.suptitle(directory)
    plt.legend(loc='upper right', title = 'objectives')
    plt.show()

    