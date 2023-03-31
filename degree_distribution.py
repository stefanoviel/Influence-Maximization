from src.load import read_graph
from influence_maximization import get_communities
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analysis(path): 
    print(path)
    graph = read_graph('graphs_downscaled/' + path)
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
    print('average top 10:', np.mean(degrees_values[:10]))

    com = pd.read_csv('graph_communities/' + path.replace('txt', 'csv'))
    com = com.groupby('comm')['node'].apply(list).tolist()
    num_elems = [len(i) for i in com]

    print('min:', min(num_elems))
    print('max:', max(num_elems))
    print('num:', len(num_elems))

    print('\n')

    return num_elems
    # plt.bar(x = list(range(len(num_elems))), height = num_elems)
    # plt.title(path)
    # plt.show()



data1 = analysis('deezerEU_4.txt')
data2 = analysis('facebook_combined_4.txt')
data3 = analysis('fb_politician_4.txt')
data4 = analysis('pgp_4.txt')

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot data on each axis
ax[0, 0].plot(range(len(data1)), data1)
ax[0, 1].plot(range(len(data2)), data2)
ax[1, 0].plot(range(len(data3)), data3)
ax[1, 1].plot(range(len(data4)), data4)

# Set axis labels and titles
ax[0, 0].set_ylabel('elements per community')
ax[0, 0].set_title('deezerEU_4')
ax[0, 1].set_title('facebook_combined_4')
ax[1, 0].set_ylabel('elements per community')
ax[1, 0].set_xlabel('communities')
ax[1, 0].set_title('fb_politician_4')
ax[1, 1].set_xlabel('communities')
ax[1, 1].set_title('pgp_4')

# Set x-tick labels

# Show the plot
plt.show()