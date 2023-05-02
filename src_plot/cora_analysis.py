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

if __name__ == "__main__": 
        

    labels = pd.read_csv('labelled_graphs/cora/cora_labels_proc', sep='\t', header=None, names=['node', 'label'])
    labels.label = labels.label.apply(lambda x : x.replace('_', ' '))

    label_distribution = False

    if label_distribution: 

        labels_dist = (labels.groupby(by = ['label']).count()/len(labels)) * 100
        print(labels_dist)
        labels_dist = labels_dist.to_dict()['node']
        print(labels_dist)


        names = list(labels_dist.keys())
        values = list(labels_dist.values()) 

        fig = plt.figure()
        plt.bar(range(len(labels_dist)), values, tick_label=names)
        plt.ylabel('% papers')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.25)
        fig.autofmt_xdate(rotation=30)
        plt.show()

    else: 

        labels_dist = labels.groupby(by = ['label']).count()
        labels_dist = labels_dist.to_dict()['node']


        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        i = 0
        for fitness_funct in os.listdir('exp1_out_cora-IC-0.05'): 

            posx = [0,0,1,1]
            posy = [0,1,0,1]
            labels_counts = []
            all_solution, mean_size = extract_solutions(os.path.join('exp1_out_cora-IC-0.05', fitness_funct))
            # print(solution_nodes)
            print(fitness_funct)
            df = pd.DataFrame()
            for solution_nodes in all_solution: 
                labels_counts = labels[labels['node'].isin(eval(solution_nodes))].label

                res = Counter(labels_counts)
                data =  dict(sorted(res.items()))
                # print(data) # normalize by the total number of papers of that type in the original network
                data = dict((k, float(data[k]/(labels_dist[k]))) for k in data)
                
                df = pd.concat([df, pd.DataFrame([data], index=[0])], ignore_index=True)
                df = df.fillna(0)
                
            
            print(df.mean())


            ax[posx[i], posy[i]].bar(range(len(df.columns)), np.array(df.mean())*100, tick_label=df.columns)
            ax[posx[i], posy[i]].set_title(fitness_funct.replace('_', ' '))
            ax[posx[i], posy[i]].set_ylim([0, ])
            fig.autofmt_xdate(rotation=45)
            i += 1


        # fig.ylabel('% percentage of papers in each category \n normalized by global percentage')
        fig.text(0.04, 0.5, "% of labels per category", va='center', rotation='vertical')
        plt.show()