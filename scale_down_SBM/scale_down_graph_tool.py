import networkx as nx
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


from graph_tool.all import *
scale = 2
resolution = 6

filename = "lastf_asia.txt"
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


i = 1
for item in check:
    for node in item:
        with open('comm_ground_truth/'+name, 'a') as the_file:
            the_file.write(str(node) + ","+ str(i)+ "\n")
    i +=1

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
    #all_edges[i][i] = float((2*edges)/(nodes*(nodes-1)))
    avg =  edges / nodes
    all_edges[i][i] = (((nodes / scale) * avg)) *2 #* 2
    all_edges[i][i] = (((nodes / scale) * avg)) * 2
    print(nodes, nodes/scale)
    print(edges,all_edges[i][i], edges/all_edges[i][i], avg)
    #all_edges[i][i] = edges
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

            avg = (edge / nodes) 
            all_edges[i][j] = (nodes / scale) * avg
            all_edges[j][i] = (nodes / scale) * avg
            #all_edges[i][j] = edge
            #all_edges[j][i] = edge
            #if  all_edges[j][i] > 1:
            #    all_edges[j][i] = 1
            #    all_edges[i][j] = 1     
            n = n -2
        if n == 0:
            break
    if n == 0:
        break

sizes = []
for item in check:
    sizes.append(int(len(item)/scale))

x = []
start = 0
for i in range(len(sizes)):
    k = i
    xx= []
    for t in range(start,sizes[i]+start):
        xx.append(t)
    x.append(xx)
    start=  start+sizes[i]
for i in range(len(sizes)):
    print("Community {0} has {1} elements".format(i+1,sizes[i]))

print(all_edges)
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
##------------------------------------------------------------------------------------------------------------


my_degree_function = G.degree


degree = {}
for item in G:
    degree[item] = my_degree_function(item)
print(degree)

maxx = []
scale_degree = {}
for key, value in degree.items():
    scale_degree[key] = int(value/scale) if int(value/scale)>1 else 1
    maxx.append(scale_degree[key])



i = 0
comm_degree = {}
default_degree = {}
for item in check:
    for node in item:
        print('Node {0}, Comm {1}'.format(node,i))
        try:
            tt = comm_degree[i]
            dd = default_degree[i]
        except:
            tt = []
            dd = []
        tt.append(scale_degree[node])
        comm_degree[i] = tt
        dd.append(degree[node])
        default_degree[i] = dd   
    i +=1
     





##------------------------------------------------------------------------------------------------------------


gt_degree = {}

out = []
nodes = []
real_degree = {}



for i in range(0,len(sizes)):
    t = []
    import copy, random

    list_degree = copy.deepcopy(comm_degree[i])
    print('Community {0}'.format(i+1))
    print("--------")

    k = sizes[i]
    probability = []
    items = []
    for item in set(list_degree):
        probability.append(list_degree.count(item) / len(list_degree))
        items.append(item)

    current_best = None
    for z in range(10):
        mm_list = random.choices(items, probability,k=k)
        if current_best == None:
            current_best = max(mm_list)
            final_list = mm_list
        else:
            if current_best < max(mm_list):
                current_best = max(mm_list)
                final_list = mm_list     
        print(current_best, max(mm_list)) 

    print(current_best)
    

    #print(final_list)
    print(max(final_list))
    for item in final_list:
        nodes.append(i)
        t.append(item)
        out.append(item)
    real_degree[i] = t


print(nodes)
print(out)
print(len(nodes), len(out))

print(max(out))
for i in range(0,len(check)):
    mean = default_degree[i]
    mean_1 = comm_degree[i]
    mean_real = real_degree[i]
    import seaborn as sns
    fig, axs = plt.subplots(ncols=3)
    sns.distplot(mean, hist=True, kde=True, 
                 color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4},ax=axs[0])
    sns.distplot(mean, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 3},ax=axs[0])    
    sns.distplot(mean_1, hist=True, kde=True, 
                 color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4},ax=axs[1])
    sns.distplot(mean_1, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 3},ax=axs[1])
   
    sns.distplot(mean_real, hist=True, kde=True, 
                 color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4},ax=axs[2])
    sns.distplot(mean_real, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 3},ax=axs[2]) 

    axs[0].set_title('Original Graph')  
    axs[0].set_xlabel('In\out Degree')
    axs[1].set_title('50% Graph')
    axs[1].set_xlabel('In\out Degree')
    axs[2].set_title('50% Graph')
    axs[2].set_xlabel('In\out Degree')
    #plt.show()


from scipy import sparse
m = np.array(all_edges)
m=sparse.csr_matrix(m)
print(m)

#mrs, out_teta = graph_tool.generation.solve_sbm_fugacities(nodes, m, out_degs=out, in_degs=None, multigraph=False, self_loops=False, epsilon=1e-08, iter_solve=True, max_iter=0, min_args={}, root_args={}, verbose=False)
#print(mrs)
#print(out_teta)

g = graph_tool.generation.generate_sbm(nodes, m, out_degs=out, in_degs=None, directed=False, micro_ers=True, micro_degs=False)
# sum = 0
# for v in g.vertices():
#     sum +=1
# print(sum)
# graph_tool.stats.remove_self_loops(g)
##g = graph_tool.generation.generate_maxent_sbm(nodes, mrs, out_teta, in_theta=None, directed=False, multigraph=False, self_loops=False)
#graph_draw(g, vertex_text=g.vertex_index, output="two-nodes.pdf")
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

# text = []
# for u,v in g.edges():
#     f = "{0} {1}".format(u,v)
#     text.append(f) 

# name = name.replace(".txt","")

with open("scale_graphs/"+str(name)+"_"+"True-"+str(scale)+".txt", "w") as outfile:
        outfile.write("\n".join(text))
    
print("ok")