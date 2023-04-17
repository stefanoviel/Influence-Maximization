import sys
import pandas as pd
import seaborn as sns
import networkx as nx
import os
sys.path.append("/home/stefano/Documents/UNITN/tesi/Influence-Maximization")
from src.load import read_graph


for path in ['facebook_combined_4','fb_politician_4','pgp_4','deezerEU_4']: 
    graph = read_graph('graphs_downscaled/' + path + '.txt')
    print(path)
    # print(len(list(nx.connected_components(graph))))
    # for comp in nx.connected_components(graph): 
    #     print(len(comp) / graph.number_of_nodes())

    com = pd.read_csv('graph_communities/' + path + '.csv')
    # print(len(com))
    com = com.groupby('comm')['node'].apply(list).tolist()     

    nodes = list(graph.nodes)
    nodes.sort()
    # print(nodes[-1])
    # print(len(nodes))

    # break 
    edges_out_comm = 0
    for c in com: 
        for node in c: 
            try: 
                for neighbor in graph.neighbors(node): 
                    if neighbor not in c: 
                        edges_out_comm += 1

            except nx.exception.NetworkXError: 
                pass

    print(edges_out_comm / graph.number_of_edges())
