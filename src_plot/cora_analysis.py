# which are the most present labels based on the objective we are optimizing for
from collections import Counter
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
sys.path.append("/home/stefano/Documents/UNITN/tesi/Influence-Maximization")
# print(sys.path)
from src.load import read_graph
import numpy as np
import pandas as pd


def extract_solutions(path):
    solution_nodes = []
    seed_set_size = []
    for file in os.listdir(path): 
        if 'hv' not in file and 'time' not in file: 
            solution = pd.read_csv(os.path.join(path, file))
            solution_nodes = solution_nodes + solution.nodes.to_list()
            seed_set_size = seed_set_size + [len(eval(s)) for s in  solution.nodes.to_list()]
    # print(seed_set_size)
    return solution_nodes, np.mean(seed_set_size)

labels = pd.read_csv('labelled_graphs/cora/cora_labels_proc', sep='\t', header=None, names=['node', 'label'])
labels.label = labels.label.apply(lambda x : x.replace('_', ' '))

labels_dist = labels.groupby(by = ['label']).count()  #/len(labels))['node'] * 100
labels_dist = labels_dist.to_dict()['node']
print(labels_dist)

names = list(labels_dist.keys())
values = list(labels_dist.values())

fig = plt.figure()
plt.bar(range(len(labels_dist)), values, tick_label=names)
plt.ylabel('# papers')
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.25)
fig.autofmt_xdate(rotation=30)
plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# i = 0
# for fitness_funct in os.listdir('exp1_out_cora-IC-0.05'): 

#     posx = [0,0,1,1]
#     posy = [0,1,0,1]
#     labels_counts = []
#     all_solution, mean_size = extract_solutions(os.path.join('exp1_out_cora-IC-0.05', fitness_funct))
#     # print(solution_nodes)
#     print(fitness_funct)
#     for solution_nodes in all_solution: 
#         for solution in eval(solution_nodes): 
#             labels_counts.append(labels.loc[labels['node'] == solution].iloc[0].label)

#     res = Counter(labels_counts)
#     data =  dict(sorted(res.items()))
#     # print(data) # normalize by the total number of papers of that type in the original network
#     data = dict((k, float(data[k]/(labels_dist[k]*15))) for k in data)
#     names = list(data.keys())

#     values = list(data.values())    


#     ax[posx[i], posy[i]].bar(range(len(data)), np.array(values), tick_label=names)
#     ax[posx[i], posy[i]].set_title(fitness_funct)
#     fig.autofmt_xdate(rotation=45)
#     i += 1


# # fig.ylabel('% percentage of papers in each category \n normalized by global percentage')
# fig.text(0.04, 0.5, "normalized nodes per category", va='center', rotation='vertical')
# plt.show()