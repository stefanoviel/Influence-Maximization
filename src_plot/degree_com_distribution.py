import sys
import os
sys.path.append("/home/stefano/Documents/tesi/Influence-Maximization")
print(sys.path)
from src.load import read_graph
from influence_maximization import get_communities
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analysis(path): 
    print(path)
    graph = read_graph('graphs/' + path)
    print('number of nodes:', graph.number_of_nodes())
    print('number of edges:', graph.number_of_edges())

    degrees = graph.degree
    degrees_values = []
    for node, d in degrees: 
        degrees_values.append(d)
    
    print('mean degree', round(np.mean(degrees_values), 2))
    print('max degree', max(degrees_values))
    print('std degree', round(np.std(degrees_values), 2))
    degrees_values.sort(reverse = True)
    print('average degree top 2.5% :', np.mean(degrees_values[:10]) )

    com = pd.read_csv('graph_communities/' + path.replace('txt', 'csv'))
    com = com.groupby('comm')['node'].apply(list).tolist()     
    elems_per_community = [len(i) for i in com]

    print('min:', min(elems_per_community))
    print('max:', max(elems_per_community))
    print('num:', len(elems_per_community))

    print('\n')

    return (np.array(elems_per_community) / graph.number_of_nodes())*100, ((np.array(degrees_values)) / graph.number_of_nodes())*100
    # plt.bar(x = list(range(len(num_elems))), height = num_elems)
    # plt.title(path)
    # plt.show()



com_deezer, degree_deezer = analysis('cora.txt')
# data2, degrees_distribution2 = analysis('facebook_combined_4.txt')
# data3, degrees_distribution3 = analysis('fb_politician_4.txt')
# data4, degrees_distribution4 = analysis('pgp_4.txt')

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# # Plot data on each axis
# ax[0, 0].bar(range(len(com_deezer)), com_deezer)
# ax[0, 1].bar(range(len(data2)), data2)
# ax[1, 0].bar(range(len(data3)), data3)
# ax[1, 1].bar(range(len(data4)), data4)

# # Set axis labels and titles
# ax[0, 0].set_ylabel('%nodes per community')
# ax[0, 0].set_title('deezerEU_4')
# ax[0, 1].set_title('facebook_combined_4')
# ax[1, 0].set_ylabel('%nodes per community')
# ax[1, 0].set_xlabel('communities')
# ax[1, 0].set_title('fb_politician_4')
# ax[1, 1].set_xlabel('communities')
# ax[1, 1].set_title('pgp_4')

# # plt.show()

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))


# ax[0, 0].bar(range(len(degree_deezer)), degree_deezer)
# ax[0, 1].bar(range(len(degrees_distribution2)), degrees_distribution2)
# ax[1, 0].bar(range(len(degrees_distribution3)), degrees_distribution3)
# ax[1, 1].bar(range(len(degrees_distribution4)), degrees_distribution4)

# # Set axis labels and titles
# ax[0, 0].set_ylabel('%neighbors per node')
# ax[0, 0].set_title('deezerEU_4')
# ax[0, 1].set_title('facebook_combined_4')
# ax[1, 0].set_ylabel('%neighbors per node')
# ax[1, 0].set_xlabel('nodes')
# ax[1, 0].set_title('fb_politician_4')
# ax[1, 1].set_xlabel('nodes')
# ax[1, 1].set_title('pgp_4')

# plt.show()