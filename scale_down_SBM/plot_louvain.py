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

scale_vector = [1.33,1.5,2,3,4,5]
no_simulations = 10
X = 100

def SBM(GR,check,s,scale_p):

    sum = 0
    check_ok = []

    for item in check:
        sum = sum + len(item)

        if len(item) > 2*scale_p:
            check_ok.append(item)



    check = check_ok



    list_edges = []
    for i in range(len(check)):
        edge = 0
        for k in range(0,len(check[i])-1):
            for j in range(k+1,len(check[i])):
                if GR.has_edge(check[i][k],check[i][j]) == True:
                    edge = edge + 1
        
        list_edges.append(edge)



    all_edges =  [[0 for x in range(len(check))] for y in range(len(check))] 
    for i in range(len(list_edges)):
        nodes = len(check[i])
        edges = list_edges[i]
        all_edges[i][i] = float((2*edges)/(nodes*(nodes-1)))

    n = (len(check) * len(check)) - len(check)

    for i in range(len(check)):
        for j in range(i+1,len(check)):

            edge=0
            if i != j:
                for node1 in check[i]:
                    for node2 in check[j]:
                        if GR.has_edge(node1,node2) == True:
                                edge = edge + 1 
                
                nodes = len(check[i]) + len(check[j])
        
                all_edges[i][j] = float((2*edge)/(nodes*(nodes-1)))
                all_edges[j][i] = float((2*edge)/(nodes*(nodes-1)))
                if  all_edges[j][i] >1:
                    all_edges[j][i] = 1
                    all_edges[i][j] = 1     
                n = n -2
            if n == 0:
                break
        if n == 0:
            break


    sizes = []
    for item in check:
        sizes.append(int(len(item)/scale_p))


    g = nx.stochastic_block_model(sizes, all_edges, seed=0)

    den = (2*g.number_of_edges())/ (g.number_of_nodes()*(g.number_of_nodes()-1))
    #print(den)
    with open('scale_results_csv/'+name, 'a') as the_file:
        the_file.write(str(den) + ","+ str(len(check)) +"," +str(s) +"\n")
    
    return den


filename = "graphs/ego-twitter.txt"


G = read_graph(filename)
G = G.to_undirected()

print(nx.info(G))
original_density = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
print("Density --> {0}".format(original_density))

N = G.number_of_nodes()

no_simulations = 10
X = 100

for scale in scale_vector:
    scale = int(scale)
    resolution = 1
    name = (os.path.basename(filename))
    name = name + "_" + str(scale)
    for i in range(int(X)):
        comm_values = []
        size_values = []
        list_check = []
        for n in range(no_simulations):
            print(resolution)
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
        check = list_check[index]
        density = SBM(G,check,size_values[index],scale)
        resolution = round(resolution +1,2)

