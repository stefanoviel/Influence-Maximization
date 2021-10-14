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

resolution = 1
no_simulations = 5

filename = "graphs/facebook_combined.txt"
name = (os.path.basename(filename))
G = load.read_graph(filename)
G = G.to_undirected()

print(nx.info(G))

N = G.number_of_nodes()
X = 100

smallest = []
communities = []
for i in range(int(X)):
    comm_values = []
    size_values = []
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
        size = []
        for k in range(len(check)):
            size.append(len(check[k]))
    
    
        comm_values.append(len(df["comm"]))
        size_values.append(min(size))
        print("Com Values {0} , Size Values {1}".format(comm_values,size_values))
    
    com_max = max(comm_values)
    index = comm_values.index(com_max)

    communities.append(com_max)
    smallest.append(size_values[index])
    resolution = round(resolution +1,2)


print(communities)
print(smallest)
import matplotlib.pyplot as plt

#x1 = [x* 0.1 for x in range(1,X+1)]
x1 = [x for x in range(1,X+1)]

print(x1)
y1 = communities

plt.plot(x1, y1, label = "communities", color="red")
plt.xlabel('r - resolution')

# Set the y axis label of the current axis.
plt.ylabel('#C -  no communities')
plt.legend()
plt.savefig(name+"r-#c.png", dpi=200)
plt.show()

y2 = smallest
plt.plot(x1, y2, label = "size", color="blue")

plt.xlabel('r - resolution')
# Set the y axis label of the current axis.
plt.ylabel('#S - Size smallest community')
# Set a title of the current axes.
# show a legend on the plot
plt.legend()
# Display a figure.
plt.savefig(name+"r-#s.png", dpi=200)

plt.show()


plt.plot(x1, y1, label = "communities", color="red")
plt.plot(x1, y2, label = "size", color="blue")

plt.xlabel('r - resolution')

# Set the y axis label of the current axis.
plt.ylabel('S - Size')
plt.legend()
plt.savefig(name+"r-#C#s.png", dpi=200)

plt.show()



