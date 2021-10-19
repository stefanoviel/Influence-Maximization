import networkx as nx
import matplotlib.pylab as pl
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

scale = 4
resolution = 22

filename = "graphs/facebook_combined.txt"
name = (os.path.basename(filename))
G = read_graph(filename)
G = G.to_undirected()


print(nx.info(G))

den = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
print("Density --> {0}".format(den))
"""
Resolution is a parameter for the Louvain community detection algorithm that affects the size of the 
recovered clusters. Smaller resolutions recover smaller, and therefore a larger number of clusters, 
and conversely, larger values recover clusters containing more data points.
"""
      
partition = community_louvain.best_partition(G, resolution=resolution)

"""REDIFNE CHECK LIST HERE"""
df = pd.DataFrame()
df["nodes"] = list(partition.keys())
df["comm"] = list(partition.values()) 
df = df.groupby('comm')['nodes'].apply(list)
df = df.reset_index(name='nodes')
check = []
for i in range(max(partition.values())+1):
    check.append(df["nodes"].iloc[i])
print(df)
sum = 0
check_ok = []

for item in check:
    sum = sum + len(item)

    if len(item) > 2*scale:
        check_ok.append(item)



print("Total number of nodes after selection {0} \nCommunities before check {1} \nCommunities after check {2}".format(sum,len(check),len(check_ok)))
check = check_ok



list_edges = []
for i in range(len(check)):
    edge = 0
    for k in range(0,len(check[i])-1):
        for j in range(k+1,len(check[i])):
            if G.has_edge(check[i][k],check[i][j]) == True:
                edge = edge + 1
    
    list_edges.append(edge)


for i in range(len(check)):
    print("Community {0} has {1} elements".format(i,len(check[i])))

all_edges =  [[0 for x in range(len(check))] for y in range(len(check))] 
for i in range(len(list_edges)):
    nodes = len(check[i])
    edges = list_edges[i]
    print("Community {0} --> Edges = {1} , Nodes = {2}".format(i,edges,nodes))
    all_edges[i][i] = float((2*edges)/(nodes*(nodes-1)))

n = (len(check) * len(check)) - len(check)


for i in range(len(check)):
    print("I --> {0}".format(i))
    for j in range(i+1,len(check)):
        print("J --> {0}".format(j))

        edge=0
        if i != j:
            for node1 in check[i]:
                for node2 in check[j]:
                    if G.has_edge(node1,node2) == True:
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
    sizes.append(int(len(item)/scale))

for i in range(len(sizes)):
    print("Community {0} has {1} elements".format(i+1,sizes[i]))

print(all_edges)

g = nx.stochastic_block_model(sizes, all_edges, seed=0)
print(nx.info(g))



den = (2*g.number_of_edges())/ (g.number_of_nodes()*(g.number_of_nodes()-1))
print("Density --> {0}".format(den)) 

text = []
for u,v in g.edges():
    f = "{0} {1}".format(u,v)
    text.append(f) 

name = name.replace(".txt","")
with open("scale_graphs/"+str(name)+"_"+"scale_"+str(scale)+".txt", "w") as outfile:
        outfile.write("\n".join(text))