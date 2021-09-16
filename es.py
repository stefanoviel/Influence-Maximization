import networkx as nx
import matplotlib.pylab as pl
import numpy as np
import community 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from numpy.core.fromnumeric import size
import load
# sizes = [40,50,100,200,150]

# N=len(sizes)
# #a = np.random.rand(N, N)
# a = np.random.uniform(0.0005, 0.005, N)
# m = np.tril(a) + np.tril(a, -1).T
# m = m.tolist()
# x = 0

# ## to change the probability of edges in a community
# for i in range(len(m)):
#     new = m[i]
#     new[x] = np.random.uniform(0.5, 1, 1)
#     m[i]=new
#     x= x+1
# print(m)
# print(type(m))

# g = nx.stochastic_block_model(sizes, m, seed=0)
# position = nx.spring_layout(g)
# nx.draw(g, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=50,linewidths=1, edge_color="#C0C0C0", width=0.5)
# pl.show()

filename = "graphs/Amazon0302.txt"
G = load.read_graph(filename)
G = G.to_undirected()
partition = community.best_partition(G)

print(nx.info(G))

##add number and nodes to list for each community
check = []
check1 = True
l = []
for key, value in partition.items():
    
    if check1 == True:
        i = value
        check1=False
    
    #print(f'{key}={value}')
    if value == i:
        l.append(key)
    else:
        check.append(l)
        l = []
        i = value
        l.append(key)


check.append(l)

## Check in order to keep only the community with at least k elements
check_ok = []
for i in range(len(check)):
    if len(check[i]) > 9:
        check_ok.append(check[i])

##Check Number of nodes
sum = 0
for item in check_ok:
    sum = sum + len(item)

print("Total number of nodes after selection {0} \nCommunities before check {1} \nCommunities after check {2}".format(sum,len(check),len(check_ok)))

check = check_ok
list_edges = []
for i in range(len(check)):
    edge = 0
    for k in range(0,len(check[i])):
        for j in range(0,len(check[i])):
            if check[i][k] != check[i][j]:
                if G.has_edge(check[i][k],check[i][j]) == True:
                    edge = edge + 1
    
    list_edges.append(edge/2)



all_edges =  [[0 for x in range(len(check))] for y in range(len(check))] 
for i in range(len(list_edges)):
    nodes = len(check[i])
    edges = list_edges[i]
    #print((2*edges)/(nodes*(nodes-1)))

    all_edges[i][i] = float((2*edges)/(nodes*(nodes-1)))

#print(all_edges)
#print(type(all_edges))

for i in range(len(check)):
    edge = 0
    for k in range(0,len(check[i])):
        for j in range(0,len(check)):
            if j!=i:
                for item in check[j]:
                    if G.has_edge(check[i][k],item) == True:
                        edge = edge + 1    
        
                nodes = len(check[i]) + len(check[j])
                all_edges[i][j] = float((edge)/(nodes*(nodes-1)))
                all_edges[j][i] = float((edge)/(nodes*(nodes-1)))
                #print("For k {0} and j {1} number of nodes {2} and edges {3}".format(k,j,nodes,edge))




sizes = []
for item in check:
    sizes.append(int(len(item)))

for i in range(len(sizes)):
    print("Community {0} has {1} elements".format(i+1,sizes[i]))

g = nx.stochastic_block_model(sizes, all_edges, seed=0)
position = nx.spring_layout(g)

#nx.draw(g, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=50,linewidths=1, edge_color="#C0C0C0", width=0.5)
#pl.show()
G = g
partition = community.best_partition(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

nx.draw_networkx_nodes(G, position, partition.keys(), node_size=50,
                        cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, position, alpha=0.5)

plt.savefig("Graph.png", format="PNG")
plt.show()
    # draw the graph



# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=50,
#                        cmap=cmap, node_color=list(partition.values()))


# #nx.draw(g, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=50,linewidths=1, edge_color="#C0C0C0", width=0.5)

# nx.draw_networkx_edges(G, pos, alpha=0.5)

# plt.show()
