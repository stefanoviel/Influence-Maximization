import sys
import seaborn as sns
import os
sys.path.append("/home/stefano/Documents/UNITN/tesi/Influence-Maximization")
# print(sys.path)
from src.load import read_graph
from collections import Counter
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def analysis(path): 
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    graph = read_graph('graphs_downscaled/' + path + '.txt')
    posx = [0,0,1,1]
    posy = [0,1,0,1]
    exp_path = 'exp1_out_' + path + '-IC-0.05'
    i = 0
    print(path)
    avg_node_degree = []
    avg_seed_size = []
    for fitness_funct in os.listdir(exp_path): 
        if 'seed' not in fitness_funct: 
            continue
        
        print(fitness_funct)
        solution_runs, mean_seed_size = extract_solutions(os.path.join(exp_path, fitness_funct))

        print('mean seed size', round(mean_seed_size/graph.number_of_nodes() * 100, 3))

        communities = pd.read_csv('graph_communities/' + path +'.csv')
        communities = communities.groupby('comm')['node'].apply(list).tolist()
        number_elements_solution_com = []
        solution_nodes_degree = []
        tot_nodes = 0
        for solution_nodes in solution_runs: 
            for solution in eval(solution_nodes): 
                solution_nodes_degree.append(graph.degree[solution])
                for com in communities: 
                    if solution in com: 
                        tot_nodes += 1
                        number_elements_solution_com.append(len(com))

        print("average solution nodes' degree", np.average(solution_nodes_degree), mean_seed_size)
        avg_node_degree.append(np.sum(solution_nodes_degree))
        avg_seed_size.append(round(mean_seed_size/graph.number_of_nodes() * 100, 3))
        # print()
    #     res = Counter(number_elements_solution_com)
    #     data =  dict(sorted(res.items()))
    #     names = list(data.keys())
    #     values = list(data.values())

    #     ax[posx[i], posy[i]].bar(range(len(data)), values, tick_label=names)
    #     ax[posx[i], posy[i]].set_title(fitness_funct)
    #     fig.autofmt_xdate(rotation=-90)
    #     i += 1

    # plt.show()

    return (np.array(avg_node_degree)/ graph.number_of_edges() ), np.array(avg_seed_size)

heat = []

node_degree1, seed_size1 = analysis('facebook_combined_4')
node_degree2, seed_size2 = analysis('fb_politician_4')
node_degree3, seed_size3 = analysis('pgp_4')
node_degree4, seed_size4 = analysis('deezerEU_4')
 


# heat1 = pd.DataFrame([node_degree1, node_degree2, node_degree3, node_degree4], columns=['I.S.T', 'I.S.C.T', 'I.S.C', 'I.S.'], index=['fb_combined', 'fb_politician', 'pgp', 'deezerEU'])
heat2 = pd.DataFrame([seed_size1, seed_size2, seed_size3, seed_size4], columns=['I.S.T', 'I.S.C.T', 'I.S.C', 'I.S.'], index=['fb_combined', 'fb_politician', 'pgp', 'deezerEU'])

# f, axes = plt.subplots(1, 2)

# sns.heatmap(heat1, annot=True, ax=axes[0])
# sns.heatmap(heat2, annot=True, ax=axes[1])

sns.heatmap(heat2, annot = True)


plt.show()