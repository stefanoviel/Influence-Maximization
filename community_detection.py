import os
import pandas as pd
import sys

import leidenalg as la

sys.path.insert(0, '')
from src.load import read_graph
from downscaling import from_networkx_to_igraph, leiden_algorithm, save_groud_truth_communities

network = 'facebook_combined_4'

filename = "graphs_downscaled/" + network + '.txt'
name = (os.path.basename(filename))
G = read_graph(filename)
G = G.to_undirected()
nodes = list(G.nodes)
nodes.sort()
print(nodes)

# R = from_networkx_to_igraph(G)

# communities = leiden_algorithm(R)
# reformat_com = []
# for community_num, community in enumerate(communities): 
#     for node in community: 
#         reformat_com.append([community_num, node])


# df = pd.DataFrame(reformat_com)
# df.columns = ['comm', 'node']
# df.to_csv('graph_communities/new/' + network + '.csv', sep=',', index=False)