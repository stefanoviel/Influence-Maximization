import networkx as nx
import matplotlib.pylab as pl
import numpy as np
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from numpy.lib.function_base import append
import load
import pandas as pd
import os

scale = 4
resolution = 10

Question = input("Do you want real(R) or random(RA) graph?")
if Question == ("RA"):
    name = "RANDOM_Graph_SBM"
    sizes = [60,70,200,100,150]

    N=len(sizes)
    #a = np.random.rand(N, N)
    a = np.random.uniform(0.0005, 0.005, N)
    m = np.tril(a) + np.tril(a, -1).T
    m = m.tolist()
    x = 0

    ## to change the probability of edges in a community
    for i in range(len(m)):
        new = m[i]
        new[x] = np.random.uniform(0.9, 1, 1)
        m[i]=new
        x= x+1


    G = nx.stochastic_block_model(sizes, m, seed=0)
    Z = 0
    for item in sizes:
        Z =  Z + item
    for i in range(1000):
        import random
        start = random.randint(0, Z)
        end = random.randint(0, Z)
        try:
            print('Shortest Path {0} between {1}, {2}--> {3} -- LEN {4}'.format(i,start,end,nx.shortest_path(G, source=start, target=end, weight=None, method='dijkstra'),len(nx.shortest_path(G, source=start, target=end, weight=None, method='dijkstra'))))
        except:
            pass
elif Question == ("R"):
    filename = "graphs/lastfm.txt"
    name = (os.path.basename(filename))
    G = load.read_graph(filename)
    G = G.to_undirected()


print(nx.info(G))


Question = input("Do you want print the original graph(Y/N)?")
if Question == ("Y"):      
    position = nx.spring_layout(G)

    nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
    plt.savefig("plot_graph/original-"+name+".png", dpi=1200)
    plt.cla()

"""
Resolution is a parameter for the Louvain community detection algorithm that affects the size of the 
recovered clusters. Smaller resolutions recover smaller, and therefore a larger number of clusters, 
and conversely, larger values recover clusters containing more data points.
"""
      
partition = community_louvain.best_partition(G, resolution=resolution)

##

#G1 = ig.Graph.from_networkx(G)
#partition = la.find_partition(G1, la.ModularityVertexPartition);
#print(partition)
#check = list(partition)
# print(len(check))
# check = check[:8]
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

Question = input("Do you plot the community to the original graph? (Y/N)")
if Question == ("Y"):      

    items = {}
    for i in range(len(check)):
        key = i
        for j in range(len(check[i])):
            items[check[i][j]] = key


    #print(items)
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(items.values()) + 1)
    nx.draw_networkx_nodes(G, pos, items.keys(), node_size=20,
                        cmap=cmap, node_color=list(items.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5,edge_color="#C0C0C0")
    plt.savefig("plot_graph/original-community"+name+".png", dpi=1200)
    plt.cla()


list_edges = []
for i in range(len(check)):
    edge = 0
    for k in range(0,len(check[i])-1):
        for j in range(k+1,len(check[i])):
            if check[i][k] != check[i][j]:
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

# for i in range(len(check)):
#     edge = 0
#     for k in range(0,len(check[i])):
#         for j in range(0,len(check)):
#             if j!=i:
#                 for item in check[j]:
#                     if G.has_edge(check[i][k],item) == True:
#                         edge = edge + 1    
        
            
#                 nodes = len(check[i]) + len(check[j])
#                 all_edges[i][j] = float((edge)/(nodes*(nodes-1)))
#                 all_edges[j][i] = float((edge)/(nodes*(nodes-1)))
#                 if  all_edges[j][i] >1:
#                     all_edges[j][i] = 1
#                     all_edges[i][j] = 1
                #print("For k {0} and j {1} number of nodes {2} and edges {3}".format(k,j,nodes,edge))

for i in range(len(check)):
    for j in range(i+1,len(check)):
        edge=0
        if i != j:
            for node1 in check[i]:
                for node2 in check[j]:
                    if G.has_edge(node1,node2) == True:
                             edge = edge + 1 

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

position = nx.spring_layout(g)


Question = input("Do you want print the SBM graph(Y/N)?")
if Question == ("Y"):      
    position = nx.spring_layout(g)

    nx.draw(g, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
    plt.savefig("plot_graph/SBM-"+name+"_"+ str(scale)+".png", dpi=1200)
    plt.cla()

Question = input("Do you plot the community to the SBM graph? (Y/N)")
if Question == ("Y"):      
    
    partition = community_louvain.best_partition(g, resolution=resolution)

    loop = True
    while loop:
        """REDIFNE CHECK LIST HERE"""
        partition = community_louvain.best_partition(g, resolution=resolution)
        df = pd.DataFrame()
        df["nodes"] = list(partition.keys())
        df["comm"] = list(partition.values()) 
        df = df.groupby('comm')['nodes'].apply(list)
        df = df.reset_index(name='nodes')
        check_1 = []
        for i in range(max(partition.values())+1):
            check_1.append(df["nodes"].iloc[i])
        items = {}
        for i in range(len(check_1)):
            key = i
            for j in range(len(check_1[i])):
                items[check_1[i][j]] = key

        print("Resolution {0} \nOriginal Community {1}\nCommunities SBM {2}".format(resolution, len(check), len(check_1)))

        if len(check_1) == len(check):
            loop = False
        elif len(check_1) > len(check):
            resolution = resolution + 1 
        elif len(check_1) < len(check):
            resolution = resolution - 1

    #print(items)
    pos = nx.spring_layout(g)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(items.values()) + 1)
    nx.draw_networkx_nodes(g, pos, items.keys(), node_size=40,
                        cmap=cmap, node_color=list(items.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.5,edge_color="#C0C0C0")
    plt.savefig("plot_graph/SBM-community"+name+".png", dpi=1200)

text = []
for u,v in g.edges():
    f = "{0} {1}".format(u,v)
    text.append(f) 
with open("SBM-Graph/graph_"+str(name)+"_"+"scale_"+str(scale)+".txt", "w") as outfile:
        outfile.write("\n".join(text))
