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
import random

name = "SBM"

max_nodes = 2000
min_size = int(max_nodes * 0.02)
max_size = int(max_nodes * 0.25)
sizes = []
while (max_nodes > 0):
    size = random.randint(min_size, max_size)
    sizes.append(size)
    max_nodes -= size

print(len(sizes))
N=len(sizes)
a = np.random.uniform(0.00005, 0.0005, N)
m = np.tril(a) + np.tril(a, -1).T
m = m.tolist()
x = 0

## to change the probability of edges in a community
for i in range(len(m)):
    new = m[i]
    new[x] = np.random.uniform(0.05,0.1,1)
    m[i]=new
    x= x+1


G = nx.stochastic_block_model(sizes, m, seed=0)



     
position = nx.spring_layout(G)

nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
plt.savefig("original-"+name+".png", dpi=300)
plt.cla()

text = []
for u,v in G.edges():
    f = "{0} {1}".format(u,v)
    text.append(f) 
with open("graph_"+str(name)+".txt", "w") as outfile:
        outfile.write("\n".join(text))
print(nx.info(G))
original_density = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
print("Density --> {0}".format(original_density))