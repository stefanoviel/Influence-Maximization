from os import get_terminal_size
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
from graph_tool.all import *
'''
import graph_tool.all as gt
g = gt.collection.data["polblogs"]
g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
g = gt.Graph(g, prune=True)
state = gt.minimize_blockmodel_dl(g)
print(state.b.a)
print(type(state.b.a))
print(gt.adjacency(state.get_bg(),state.get_ers()).T)
exit(0)
u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(),
                                               state.get_ers()).T,
                       g.degree_property_map("out").a,
                       g.degree_property_map("in").a, directed=True)
gt.graph_draw(g, g.vp.pos, output="polblogs-sbm.pdf")
gt.graph_draw(u, u.own_property(g.vp.pos), output="polblogs-sbm-generated.pdf")
exit(0)
'''


name = "SBM"

max_nodes = 100
min_size = int(max_nodes * 0.1)
max_size = int(max_nodes * 0.3)
sizes = []
while (max_nodes > 0):
    size = random.randint(min_size, max_size)
    sizes.append(size)
    max_nodes -= size

print(len(sizes))
N=len(sizes)
#a = np.random.uniform(0.000005, 0.0005, N)
a = np.random.uniform(50, 100, N)
m = np.tril(a) + np.tril(a, -1).T
m = m.tolist()
x = 0

## to change the probability of edges in a community
for i in range(len(m)):
    new = m[i]
    new[x] = float(np.random.uniform(5000,1000,1))
    m[i]=new
    x= x+1

from scipy import sparse
m = np.array(m)
m=sparse.csr_matrix(m)
print(m)

x = []
start = 0
for i in range(len(sizes)):
    k = i
    xx= []
    for t in range(start,sizes[i]+start):
        xx.append(t)
    x.append(xx)
    start=  start+sizes[i]


nodes =[]
out = []
i = 0
for item in x:
    k = i
    for node in item:
        nodes.append(k)
        #nodes.append(node)
        #out.append(node)
        out.append(int(np.random.uniform(1,10,1)))
    i = i+1

print(nodes)
print(len(nodes))
#sizes = [0,1,0,2,0,3,0,4,0,5,0,6,0,7,1,8,1,8,1,8,2,9,2,10,3,11,4,12,5,15,6,16]


max_nodes = max(nodes)

mrs, out_teta = graph_tool.generation.solve_sbm_fugacities(nodes, m, out_degs=out, in_degs=None, multigraph=False, self_loops=False, epsilon=1e-08, iter_solve=True, max_iter=0, min_args={}, root_args={}, verbose=False)
print(mrs)
print(out_teta)
g = graph_tool.generation.generate_maxent_sbm(nodes, mrs, out_teta, in_theta=None, directed=False, multigraph=False, self_loops=False)
#g = graph_tool.generation.generate_sbm(nodes, m, out_degs=out, in_degs=None, directed=False, micro_ers=False, micro_degs=False)
edges = 0


print(edges)
graph_tool.stats.remove_self_loops(g)

for e in g.edges():
    edges += 1
    t = str(e)
    t = t.replace("(","")
    t = t.replace(")","")
    t = t.replace(",","")
    nodes = list(t.split(" "))
    #print(nodes)
    #print(nodes[0], nodes[1])
sum = 0
for v in g.vertices():
    sum +=1
edges = 0
for e in g.edges():
    edges += 1

print(sum)
print(edges)

graph_draw(g, vertex_text=g.vertex_index, output="two-nodes.pdf")

exit(0)


print("ok")
exit(0)
position = nx.spring_layout(G)

nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
#plt.savefig("original-"+name+".png", dpi=300)
plt.show()
plt.cla()