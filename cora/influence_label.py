# for each seed set (all elements of PF in all runs)
    # run the spread process with the right probability save the resulting seed set
    # compute the distribution on these labels

from src.spread.monte_carlo_3_obj import IC_model_influenced_nodes
from src_plot.cora_analysis import extract_solutions
from src.load import read_graph 
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import random
import os


df = pd.read_csv('networks_max_times.csv')
max_time = df.cora.iloc[0]

graph = read_graph('graphs/cora.txt')

# load labels
labels = pd.read_csv('labelled_graphs/cora/cora_labels_proc', sep='\t', header=None, names=['node', 'label'])
labels.label = labels.label.apply(lambda x : x.replace('_', ' '))

# compute global label distribution
labels_dist = labels.groupby(by = ['label']).count()
labels_dist = labels_dist.to_dict()['node']



result_dir = 'exp1_out_cora-IC-0.05'
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
i = 0
for fitness_function in os.listdir(result_dir): 

    posx = [0,0,1,1]
    posy = [0,1,0,1]
    # get all seed set for all run for a certain fitness function
    all_solution, mean_size = extract_solutions(os.path.join('exp1_out_cora-IC-0.05', fitness_function))

    df_res = pd.DataFrame()
    for sol in all_solution: 
        # for each seed set find the influenced nodes
        res =  IC_model_influenced_nodes(graph, eval(sol), 0.05,random_generator= random.Random(), max_time= max_time)

        # iterate over all nodes in the graph, find if it was among the influenced labels, if true save the label
        labels_counts = labels[labels['node'].isin(res)].label
        
        res = Counter(labels_counts) # aggregate distribution of all labels
        # print(res)
        data =  dict(sorted(res.items()))
        data = dict((k, float(data[k]/(labels_dist[k]))) for k in data)  # divide by number of label in each category
        
        df_res = pd.concat([df_res, pd.DataFrame([data], index=[0])], ignore_index=True)


    df_res = df_res.fillna(0)   
    print(fitness_function)
    print(df_res.mean())

    ax[posx[i], posy[i]].bar(range(len(df_res.columns)), np.array(df_res.mean())*100, tick_label=df_res.columns)
    ax[posx[i], posy[i]].set_title(fitness_function.replace('_', ' '))
    ax[posx[i], posy[i]].set_ylim([0, 4.5])
    fig.autofmt_xdate(rotation=45)
    i += 1

fig.text(0.04, 0.5, "% of labels per category", va='center', rotation='vertical')
plt.show()
