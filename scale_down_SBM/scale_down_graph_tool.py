#basic packages
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

#graph-tool importation
from graph_tool.all import *


# local functions
sys.path.insert(0, '')
from src.load import read_graph


scale_vector = [8]

filename = "graphs/facebook_combined.txt"
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

connected_subgraphs = [G.subgraph(cc) for cc in nx.connected_components(G)]
Gcc = max(nx.connected_components(G), key=len)
G = G.subgraph(Gcc)

index = {} 
#for idx, node in enumerate(sorted(G)):
    #index[node] = idx
    #if node != idx:
    #    print(node, idx)
#G = nx.relabel_nodes(G, index)
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
t = max(scale_vector)
import leidenalg
import igraph as ig

filename = filename.replace('.txt', '')
R = ig.Graph(directed=False)
try:
    R.add_vertices(G.nodes())
    R.add_edges(G.edges())
except:
    text = []
    for e in G.edges:
        print(e)
        f = "{0} {1}".format(e[0]-1,e[1]-1)
        text.append(f) 
    with open(filename + '_correction_.txt', "w") as outfile:
            outfile.write("\n".join(text))

    G = read_graph(filename + '_correction_.txt')
    R = ig.Graph(directed=False)
    R.add_vertices(G.nodes())
    R.add_edges(G.edges())

k = 0
for i in range(len(G)):
    if R.degree(i) != G.degree(i):
        print('error')
        k +=1
        exit(0)
print(len(G), k)


part = leidenalg.find_partition(R, leidenalg.ModularityVertexPartition)
check = list(part)
print(len(check))
sum = 0
check_ok = []

for item in check:
    sum = sum + len(item)
    if len(item) >= t:
        check_ok.append(item)



i = 1
node_original = []
comm_original = []
for item in check_ok:
    for node in item:
        node_original.append(node)
        comm_original.append(i)
    i +=1

df_original = pd.DataFrame()
df_original["node"] = node_original
df_original["comm"] = comm_original

save = name.replace(".txt","")

df_original.to_csv('comm_ground_truth/'+save+'.csv',index=False, sep=",")


print("Total number of nodes after selection {0} \nCommunities before check {1} \nCommunities after check {2}".format(sum,len(check),len(check_ok)))

check = check_ok
for scale in scale_vector:
    list_edges = []
    for i in range(len(check)):
        edge = 0
        for k in range(0,len(check[i])-1):
            for j in range(k+1,len(check[i])):
                if G.has_edge(check[i][k],check[i][j]) == True:
                    edge = edge + 1
        
        list_edges.append(edge)

    all_edges =  [[0 for x in range(len(check))] for y in range(len(check))] 
    for i in range(len(list_edges)):
        nodes = len(check[i])
        edges = list_edges[i]
        avg =  edges / nodes
        all_edges[i][i] = (((nodes / scale) * avg)) * 2 #* 2
        all_edges[i][i] = (((nodes / scale) * avg)) * 2
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
                if edge > 0:
                    avg = (edge / nodes) 
                    all_edges[i][j] = int((nodes / scale) * avg)
                    all_edges[j][i] = int((nodes / scale) * avg)
                    if all_edges[i][j] < 1:
                        all_edges[i][j] = 1 
                        all_edges[j][i] = 1
                n = n -2
            if n == 0:
                break
        if n == 0:
            break

    sizes = []
    for item in check:
        sizes.append(int(len(item)/scale))
    i = 1
    start = 0
    nodes = []
    comm = []
    for item in sizes:
        for node in range(start,item+start):
            nodes.append(node)
            comm.append(i)
        start = start + item
        i +=1

    df = pd.DataFrame()
    df["node"] = nodes
    df["comm"] = comm

    save = name.replace(".txt","")

    df.to_csv('comm_ground_truth/'+save+'_'+str(scale)+'.csv',index=False, sep=",")

    print(df)

    my_degree_function = G.degree


    degree = {}
    for item in G:
        degree[item] = my_degree_function(item)
    print(degree)



    i = 0
    default_degree = {}
    for item in check:
        for node in item:
            print('Node {0}, Comm {1}'.format(node,i))
            try:
                dd = default_degree[i]
            except:
                dd = []
            dd.append(degree[node])
            default_degree[i] = dd   
        i +=1
    gt_degree = {}

    out = []
    nodes = []
    real_degree = {}



    for i in range(0,len(sizes)):
        t = []
        import copy, random

        list_degree = copy.deepcopy(default_degree[i])
        k = sizes[i]
        probability = []
        items = []
        for item in set(list_degree):
            probability.append(list_degree.count(item) / len(list_degree))
            items.append(item)

        current_best = None
        for z in range(100):
            mm_list = random.choices(items, probability,k=k)
            print(len(mm_list), len(set(mm_list)), len(list_degree), len(set(list_degree)))
            if current_best == None:
                from scipy.spatial import distance
                current_best = distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)])
                final_list = mm_list
            else:
                if current_best < distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)]):
                    current_best = distance.euclidean([np.mean(mm_list), np.std(mm_list)], [np.mean(list_degree), np.std(list_degree)])
                    final_list = mm_list  

        for item in final_list:
            nodes.append(i)
            t.append(item)
            out.append(item)
        real_degree[i] = t

    m = np.array(all_edges)
    m=sparse.csr_matrix(m)

    g = graph_tool.generation.generate_sbm(nodes, m, out_degs=out, in_degs=None, directed=False, micro_ers=True, micro_degs=False)

    sum = 0
    for v in g.vertices():
        sum +=1
    print(sum)
    edges = 0
    for e in g.edges():
        edges += 1

    text = []
    edges = 0
    n = []
    for e in g.edges():
        edges += 1
        t = str(e)
        t = t.replace("(","")
        t = t.replace(")","")
        t = t.replace(",","")
        nodes = list(t.split(" "))
        f = "{0} {1}".format(nodes[0],nodes[1])
        n.append(int(nodes[0]))
        n.append(int(nodes[1]))
        text.append(f) 



    sum = len(set(n))
    print('Density {0}'.format((2*edges)/(sum * (sum-1))))
    print(sum)
    print(max(out))

    name = name.replace('.txt',"")
    with open("scale_graphs/"+str(name)+"_"+str(scale)+".txt", "w") as outfile:
            outfile.write("\n".join(text))
        
    print("ok")