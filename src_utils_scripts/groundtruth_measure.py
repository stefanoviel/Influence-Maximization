import sys
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import degree_centrality, closeness_centrality, core_number, betweenness_centrality, katz_centrality, katz_centrality_numpy, eigenvector_centrality_numpy, current_flow_betweenness_centrality
# local libraries
sys.path.insert(0, '')
from src.load import read_graph

def read_arguments():
    """
	Parameters for the upscaling process process.
	"""
    parser = argparse.ArgumentParser(
        description='Upscaling algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--graph', default='facebook_combined',
                        choices=['facebook_combined', 'fb_politician',
                                 'deezerEU', 'fb_org', 'fb-pages-public-figuree',
                                 'pgp', 'soc-gemsec', 'soc-brightkite'],
                        help='Graph name')

    # Upscaling Parameters
    parser.add_argument('--s', type=int, default=4,
                        help='Scaling factor')  
    parser.add_argument('--measure', default='page_rank',
                        choices=['two-hop','page_rank', 'degree_centrality',
                                'katz_centrality','betweenness', 'closeness', 
                                'eigenvector_centrality', 'core'])
    args = parser.parse_args()
    args = vars(args)

    return args

#-----------------------------------------------------------------------------#

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
    data["measure"] = centr
    data = data.sort_values(by='measure', ascending=False)
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


if __name__ == '__main__':
    args = read_arguments() 
    if args["s"] != None:
        filename = "graphs_downscaled/{0}_{1}.txt".format(args["graph"],args["s"])
        filename_communities = "graph_communities/{0}.csv".format(args["graph"])
    else:
        filename = "graphs_downscaled/{0}_{1}.txt".format(args["graph"],args["s"])
        filename_communities = "graph_communities/{0}.csv".format(args["graph"])
    
    df = get_table(filename, filename_communities, args["measure"])
    df.to_csv('experiments_upscaling/measure_groundtruth/{0}-{1}-{2}.csv'.format(args["graph"], args["s"], args["measure"]), index=False)
