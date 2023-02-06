import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

for subd in os.listdir('result_comparison'): 

    df_avg = pd.read_csv('result_comparison/'+ subd + '/avg_hv.csv', index_col='fitness_function')
    df_std = pd.read_csv('result_comparison/'+ subd + '/std_hv.csv', index_col='fitness_function')

    # df.drop('fitness_function', inplace= True, axis = 1)
    df_avg = df_avg.iloc[:, :4]
    df_std = df_std.iloc[:, :4]
    df_avg = df_avg.loc[:, ~df_avg.columns.str.contains('^Unnamed')]
    df_std = df_std.loc[:, ~df_std.columns.str.contains('^Unnamed')]

    flat_list = [item for sublist in df_std.values.tolist() for item in sublist]

    scaled_df = minmax_scale(df_avg)
    x_axis_labels = [ 'hv_influence_seed', 'hv_influence_seedSize_time','hv_influence_seedSize_communities'] # labels for x-axis
    y_axis_labels = ['influence_seed', 'influence_seed_communties', 'influence_seed_time', 'influence_seed_communties_time'] # labels for y-axis

    # create seabvorn heatmap with required labels
    sns.set(rc={'figure.figsize':(14,8)})
    # df = df.round(4)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(scaled_df,  annot = df_avg, fmt='.3f', linewidths=.5, cbar=False, xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for t, std_value in zip(ax.texts, flat_list): 
        t.set_text(t.get_text() + "\n(" + str(round(std_value, 5)) + ")")

    plt.title(subd)
    plt.tight_layout()
    plt.savefig('result_comparison/'+ subd + '/avg.png',  bbox_inches='tight', dpi=300)
    plt.show()
    # break