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
import load
import pandas as pd
import os
#import igraph as ig
#import leidenalg as la
import os

resolution = 0.1
X = 100
filename = "/Users/elia/Desktop/Influence-Maximization/graphs/facebook_combined.txt"
name = (os.path.basename(filename))
G = load.read_graph(filename)
G = G.to_undirected()


print(nx.info(G))

N = G.number_of_nodes()

smallest = []
communities = []
for i in range(int(X)):
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

    size = []
    for k in range(len(check)):
        size.append(len(check[k]))

    for j in range(len(check)):
        size.append(len(check[j]))

    # if i != 0:
    #     if len(check) <= communities[len(communities)-1]:
    #         communities.append(len(check))
    #         smallest.append(min(size))
    #         resolution = resolution +1
    #     else:
    #         print("Weird case, #C {0}, size {1}".format(len(check), min(size)))
    #         i = i -1
    # else:
    communities.append(len(check))
    smallest.append(min(size))
    resolution = round(resolution +0.1,2)


print(communities)
print(smallest)
import matplotlib.pyplot as plt

x1 = [x* 0.1 for x in range(1,X+1)]
print(x1)
y1 = communities

plt.plot(x1, y1, label = "communities", color="red")
plt.xlabel('x - axis')

# Set the y axis label of the current axis.
plt.ylabel('y - axis')
plt.legend()
plt.show()

y2 = smallest
plt.plot(x1, y2, label = "size", color="blue")

plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


plt.plot(x1, y1, label = "communities", color="red")
plt.plot(x1, y2, label = "size", color="blue")

plt.xlabel('x - axis')

# Set the y axis label of the current axis.
plt.ylabel('y - axis')
plt.legend()
plt.show()



