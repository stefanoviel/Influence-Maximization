import os
import sys
import copy
import random
import logging
import argparse
import leidenalg
import numpy as np 
import igraph as ig
import pandas as pd
import networkx as nx
from scipy import sparse

# graph-tool importation. 
# For informat about installation please refere to the official website https://graph-tool.skewed.de 
# from graph_tool.all import *

# local libraries
sys.path.insert(0, '')
from src.load import read_graph

def read_arguments():
    """
	Parameters for the downscaling process.
	"""
    parser = argparse.ArgumentParser(
        description='Downscaling algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--graph', default='facebook_combined',
                        choices=['facebook_combined', 'fb_politician',
                                 'deezerEU', 'fb_org', 'fb-pages-public-figuree',
                                 'pgp', 'soc-gemsec', 'soc-brightkite'],
                        help='Graph name')
    parser.add_argument('--s', type=int, default=10,
                        help='Scaling factor')  
    args = parser.parse_args()
    args = vars(args)

    return args
#----------------------------------------------------------------------------------------------------------------#
def largest_component(G):
    Gcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(Gcc)
    return G
def from_networkx_to_igraph(G):
    R = ig.Graph(directed=False)
    R.add_vertices(list(G.nodes()))
    R.add_edges(list(G.edges()))
    return R
def leiden_algorithm(G):
    communities = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    communities = list(communities)
    return communities
def save_groud_truth_communities(communities_threshold):
    i = 1
    node_original = []
    comm_original = []
    for item in communities_threshold:
        for node in item:
            node_original.append(node)
            comm_original.append(i)
        i +=1
    df = pd.DataFrame()
    df["node"] = node_original
    df["comm"] = comm_original
    save = name.replace(".txt","")
    try:
        df.to_csv('graph_communities/'+save+'.csv',index=False, sep=",")
    except:
        df.to_csv(save+'.csv',index=False, sep=",")

def save_groud_truth_communities_scaled(communities, args):
    sizes = []
    for item in communities:
        sizes.append(int(len(item)/args["s"]))
    i = 1 
    start = 0
    nodes = []
    comm = []
    for item in sizes:
        for node in range(start,item+start):
            nodes.append(node)
            comm.append(i)
        start = start + item
        i +=1
    df = pd.DataFrame()
    df["node"] = nodes
    df["comm"] = comm
    save = name.replace(".txt","")
    try:
        df.to_csv('graph_communities/'+save+'_'+str(args["s"])+'.csv',index=False, sep=",")
    except:
        df.to_csv(save+'_'+str(args["s"])+'.csv',index=False, sep=",")
    return sizes

def save_downscaled_graph(g,name):
    text = []
    edges = 0
    n = []
    for e in g.edges():
        edges += 1
        t = str(e)
        t = t.replace("(","")
        t = t.replace(")","")
        t = t.replace(",","")
        nodes = list(t.split(" "))
        f = "{0} {1}".format(nodes[0],nodes[1])
        n.append(int(nodes[0]))
        n.append(int(nodes[1]))
        text.append(f) 

    name = name.replace('.txt',"")
    with open("graphs_downscaled/{0}_{1}.txt".format(name, args["s"]), "w") as outfile:
            outfile.write("\n".join(text))
        
def get_simmetric_matrix_edges(G, communities):
    list_edges = []
    for i in range(len(communities)):
        edge = 0
        for k in range(0,len(communities[i])-1):
            for j in range(k+1,len(communities[i])):
                if G.has_edge(communities[i][k],communities[i][j]) == True:
                    edge = edge + 1
        
        list_edges.append(edge)

    all_edges =  [[0 for x in range(len(communities))] for y in range(len(communities))] 
    for i in range(len(list_edges)):
        nodes = len(communities[i])
        edges = list_edges[i]
        avg =  edges / nodes
        all_edges[i][i] = (((nodes / args["s"]) * avg)) * 2 # multiplied by 2 for creating the diagonal matrix needed for graph-tool graph_tool.generation.generate_sbm https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.generate_sbm
        all_edges[i][i] = (((nodes / args["s"]) * avg)) * 2
    n = (len(communities) * len(communities)) - len(communities)


    for i in range(len(communities)):
        logging.debug("I --> {0}".format(i))
        for j in range(i+1,len(communities)):
            logging.debug("J --> {0}".format(j))

            edge=0
            if i != j:
                for node1 in communities[i]:
                    for node2 in communities[j]:
                        if G.has_edge(node1,node2) == True:
                                edge = edge + 1 
                
                nodes = len(communities[i]) + len(communities[j])
                if edge > 0:
                    avg = (edge / nodes) 
                    all_edges[i][j] = int((nodes / args["s"]) * avg)
                    all_edges[j][i] = int((nodes / args["s"]) * avg)
                    if all_edges[i][j] < 1:
                        all_edges[i][j] = 1 
                        all_edges[j][i] = 1
                n = n -2
            if n == 0:
                break
        if n == 0:
            break

    m = np.array(all_edges)
    m=sparse.csr_matrix(m)
    logging.debug(m)

    return m

def preserve_degree_distribution(G, communities, sizes):
    my_degree_function = G.degree
    degree = {}
    for item in G:
        degree[item] = my_degree_function(item)
    i = 0
    default_degree = {}
    for item in communities:
        for node in item:
            logging.debug('Node {0}, Comm {1}'.format(node,i))
            try:
                dd = default_degree[i]
            except:
                dd = []
            dd.append(degree[node])
            default_degree[i] = dd   
        i +=1
    out = []
    nodes = []
    for i in range(0,len(sizes)):
        t = []
        list_degree = copy.deepcopy(default_degree[i])
        k = sizes[i]
        probability = []
        items = []
        for item in set(list_degree):
            probability.append(list_degree.count(item) / len(list_degree))
            items.append(item)

        current_best = None
        for z in range(100):
            mm_list = random.choices(items, probability,k=k)
            logging.debug(len(mm_list), len(set(mm_list)), len(list_degree), len(set(list_degree)))
            if current_best == None:
                from scipy.spatial import distance
                current_best = distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)])
                final_list = mm_list
            else:
                if current_best < distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)]):
                    current_best = distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)])
                    final_list = mm_list  

        for item in final_list:
            nodes.append(i)
            t.append(item)
            out.append(item)
    
    return nodes, out
if __name__ == '__main__':
    args = read_arguments()

    filename = "graphs/{0}.txt".format(args["graph"])
    name = (os.path.basename(filename))
    G = read_graph(filename)
    G = G.to_undirected()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(nx.classes.function.info(G))

    # to use if largest component is needed
    #G = largest_component(G)
    
    # switch network to igraph to use Leiden Algotihm (implemented only for igraph type)
    R = from_networkx_to_igraph(G)

    communities = leiden_algorithm(R)
    logging.info('Number of communities detected: {0}'.format(len(communities)))
    
    # delete communities with number of elements < scaling factor
    communities_threshold = []
    for item in communities:
        if len(item) >= args["s"]:
            communities_threshold.append(item)
    save_groud_truth_communities(communities_threshold)
    logging.info("Communities before check {0} \nCommunities after check {1}".format(len(communities),len(communities_threshold)))
    
    communities = communities_threshold
    
    matrix = get_simmetric_matrix_edges(G, communities)
    sizes = save_groud_truth_communities_scaled(communities, args)
    nodes, out = preserve_degree_distribution(G, communities, sizes)
    # g = graph_tool.generation.generate_sbm(nodes, matrix, out_degs=out, in_degs=None, directed=False, micro_ers=True, micro_degs=False)
    save_downscaled_graph(g, name)

