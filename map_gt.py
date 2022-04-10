from src.load import read_graph
import pandas as pd
import numpy as np
import os
import time
import random
import logging
import networkx as nx
import numpy as np
import pandas as pd
# local libraries
from src.load import read_graph
from networkx.algorithms import degree_centrality, closeness_centrality, core_number, betweenness_centrality
from networkx.algorithms import katz_centrality, katz_centrality_numpy, eigenvector_centrality_numpy, current_flow_betweenness_centrality
def n_neighbor(G, id, n_hop):
    node = [id]
    node_visited = set()
    neighbors= []
    
    while n_hop !=0:
        neighbors= []
        for node_id in node:
            node_visited.add(node_id)
            neighbors +=  [id for id in G.neighbors(node_id) if id not in node_visited]
        node = neighbors
        n_hop -=1
        
        if len(node) == 0 :
            return neighbors 
        
    return list(set(neighbors))

def get_table(graph_name, comm_name, measure):
    G = read_graph(filename=graph_name)

    from networkx.algorithms import degree_centrality, closeness_centrality, core_number, betweenness_centrality
    from networkx.algorithms import katz_centrality, katz_centrality_numpy, eigenvector_centrality_numpy, current_flow_betweenness_centrality
    if measure == 'page_rank':
        T = nx.pagerank(G, alpha = 0.85)
    elif measure == 'degree_centrality':
        T = degree_centrality(G)
    elif measure == 'katz_centrality':
        T = katz_centrality_numpy(G)
    elif measure == 'betweenness':
        T = betweenness_centrality(G)
    elif measure == 'eigenvector_centrality':
        T = eigenvector_centrality_numpy(G)
    elif measure == 'closeness':
        T = closeness_centrality(G)
    elif measure == 'core':
        G.remove_edges_from(nx.selfloop_edges(G))
        T = core_number(G)
    elif measure == 'two-hop':
        T = {}
        for node in G:
            T[node] = len(n_neighbor(G,node,2))


    data = pd.DataFrame()
    node = []
    centr = []

    for key,value in T.items():
        node.append(key)
        centr.append(value)


    
    
    rank = [x for x in range(1,len(node)+1)]
    data["node"] = node
    data["page_rank"] = centr



    data = data.sort_values(by='page_rank', ascending=False)
    data["overall_rank"] = rank

    community = pd.read_csv(comm_name,sep=",")

    int_df = pd.merge(data, community, how ='inner', on =['node'])
    n_comm = len(set(int_df["comm"].to_list()))
    node = []
    rank_comm = []
    len_list = []
    for i in range(n_comm):
        list_comm = int_df[int_df["comm"] == i+1]
        len_list.append(len(list_comm))
        for i in range(len(list_comm)):
            node.append(list_comm["node"].iloc[i])
            rank_comm.append(i+1)

    data_comm = pd.DataFrame()
    data_comm["rank_comm"] = rank_comm
    data_comm["node"] = node


    int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
    
    return int_df

degree_measure = ['two-hop','page_rank', 'degree_centrality','katz_centrality', 'betweenness', 'closeness', 'eigenvector_centrality', 'core']

#graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
graphs = ['fb-pages-artist']
scale_factor = 16
for i in range(len(graphs)):
    filename = "scale_graphs/{0}_{1}.txt".format(graphs[i],scale_factor)
    scale_comm = "comm_ground_truth/{0}.csv".format(graphs[i])
    for measure in degree_measure:
        print(measure , '....')
        df = get_table(filename, scale_comm, measure)
        print(df)
        df.to_csv(f'map_files/{graphs[i]}-{scale_factor}-{measure}.csv', index=False)
