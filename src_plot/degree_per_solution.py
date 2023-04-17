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
    avg_node_degree = []
    graph = read_graph('graphs_downscaled/' + path + '.txt')
    exp_path = 'exp1_out_' + path + '-IC-0.05'

    for fitness_funct in os.listdir(exp_path): 
        if 'seed' not in fitness_funct: 
            continue
        
        print(fitness_funct)
        solution_runs, mean_seed_size = extract_solutions(os.path.join(exp_path, fitness_funct))

        solution_nodes_degree = []
        for solution_nodes in solution_runs: 
            solution_tot_degree = 0
            for solution in eval(solution_nodes): 
                solution_tot_degree += graph.degree[solution]

            solution_nodes_degree.append(solution_tot_degree)
        
        # print("average solution nodes' degree", np.average(solution_nodes_degree), mean_seed_size)

        print(np.average(solution_nodes_degree))
        avg_node_degree.append(np.average(solution_nodes_degree))

    
    return np.array(avg_node_degree) / graph.number_of_edges()


node_degree1 = analysis('facebook_combined_4')
node_degree2 = analysis('fb_politician_4')
node_degree3 = analysis('pgp_4')
node_degree4 = analysis('deezerEU_4')

heat1 = pd.DataFrame([node_degree1, node_degree2, node_degree3, node_degree4], columns=['I.S.T', 'I.S.C.T', 'I.S.C', 'I.S.'], index=['fb_combined', 'fb_politician', 'pgp', 'deezerEU'])
sns.heatmap(heat1, annot = True)
plt.show()