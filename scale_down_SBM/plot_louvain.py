import community
import networkx as nx
import matplotlib.pylab as pl
from networkx.generators import small
import numpy as np
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from numpy.lib.function_base import append
import sys
sys.path.insert(0, '')
from src.load import read_graph

import pandas as pd
import os
import os





graphs = ["graphs/facebook_combined.txt","graphs/fb_org.txt","graphs/fb_politician.txt","graphs/pgp.txt","graphs/deezerEU.txt"]

for filename in graphs:
    name = os.path.basename(filename)
    name = name.replace('.txt','')
    G = read_graph(filename)
    G = G.to_undirected()

    print(nx.info(G))
    original_density = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
    print("Density --> {0}".format(original_density))

    N = G.number_of_nodes()

    no_simulations = 10
    X = 100

    C = []
    S = []
    for i in range(int(X)):
        resolution = i +1
        comm_values = []
        size_values = []
        list_check = []
        print(resolution)
        for n in range(no_simulations):
            partition = community_louvain.best_partition(G, resolution=resolution)
            """REDIFNE CHECK LIST HERE"""
            df = pd.DataFrame()
            df["nodes"] = list(partition.keys())
            df["comm"] = list(partition.values()) 
            df = df.groupby('comm')['nodes'].apply(list)
            df = df.reset_index(name='nodes')
            check = []
            for j in range(max(partition.values())+1):
                check.append(df["nodes"].iloc[j])

            list_check.append(check)
            size = []
            for k in range(len(check)):
                size.append(len(check[k]))
        
        
            comm_values.append(len(df["comm"]))
            size_values.append(min(size))
            print("Com Values {0} , Size Values {1}".format(comm_values,size_values))
            
        com_max = max(comm_values)
        index = comm_values.index(com_max)
        size_final = size_values[index]
        

        C.append(com_max)
        S.append(size_final)
        print(com_max ,size_final)    

    df = pd.DataFrame()
    df["#C"] = C
    df["min_size"] = S
    df.to_csv('scale_results_csv/'+name+'.csv', index=False, sep=",")