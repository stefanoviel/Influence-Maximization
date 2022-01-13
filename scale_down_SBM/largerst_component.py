import networkx as nx
import numpy as np
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from numpy.core.numeric import False_
from numpy.lib.function_base import append
import sys
sys.path.insert(0, '')
from src.load import read_graph
import pandas as pd
import os


scale_factor = 0.125
scale = round(1/scale_factor,3)
resolution = 1

filename = "graphs/pgp.txt"
name = (os.path.basename(filename))
G = read_graph(filename)
G = G.to_undirected()
my_degree_function = G.degree

avg_degree = []
for item in G:
    avg_degree.append(my_degree_function[item])

avg_degree = np.mean(avg_degree)
print(avg_degree)

print(nx.info(G))

den = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
print("Density --> {0}".format(den))
"""
Resolution is a parameter for the Louvain community detection algorithm that affects the size of the 
recovered clusters. Smaller resolutions recover smaller, and therefore a larger number of clusters, 
and conversely, larger values recover clusters containing more data points.
"""
# test = True
# while (test): 
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
sum = 0
check_ok = []



for item in check:
    sum = sum + len(item)

    if len(item) > scale+1:
        check_ok.append(item)



print("Total number of nodes after selection {0} \nCommunities before check {1} \nCommunities after check {2}".format(sum,len(check),len(check_ok)))

print(len(check), len(check_ok))

list_difference = []
for item in check:
  if item not in check_ok:
    list_difference.append(item)

print(list_difference)

print(G.number_of_nodes())
for comm in list_difference:
    for node in comm:
        G.remove_node(node)
print(G.number_of_nodes())


text = []
for u,v in G.edges():
    f = "{0} {1}".format(u,v)
    text.append(f) 

#name = name.replace(".txt","")

with open("graphs/prova.txt", "w") as outfile:
        outfile.write("\n".join(text))