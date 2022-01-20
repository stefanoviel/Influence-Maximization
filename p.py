from cmath import log10
from pickle import TRUE
import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from math import log, log10
name = 'graph_SBM_big'

filenames = ["scale_graphs/graph_SBM_big.txt_TRUE-8.0.txt","scale_graphs/graph_SBM_big.txt_TRUE-4.0.txt","scale_graphs/graph_SBM_big.txt_TRUE-2.0.txt","graphs/graph_SBM_big.txt"]
kk = []
for item in filenames:
    G = read_graph(item)
    #print(nx.info(G))
    #den = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
    #print("Density --> {0}".format(den))
    my_degree_function = G.degree
    mean = []
    mean_degree = []
    for item in G:
        mean.append(my_degree_function[item])
    kk.append(mean)


for item in kk:
    print(np.mean(item))


x = ['1', '1/2', '1/4', '1/8']
x = x[::-1]
print(x)

real = np.mean(kk[0])
color = ["green", 'blue','orange','red']
# plt.figure(figsize=(6, 6)) 
# i = 0
# for item in filenames:
#     G = read_graph(item)   
#     degree_freq = nx.degree_histogram(G)
#     degrees = range(len(degree_freq))
#     plt.loglog(degrees, degree_freq,'go-', color=color[i], label = str(x[i]))
#     i +=1
# plt.xlabel('Degree')
# plt.ylabel('Frequency')
# plt.legend()
# plt.savefig('degree_log.png')

# #plt.show()
# plt.cla()
# plt.close()


i=0
filenames = filenames[::-1]
x = x[::-1]
color = color[::-1]
i = 0
fig, axs = plt.subplots(4, sharex=True,figsize=(6, 6)) 
#fig.suptitle('Degree Distribution')

import collections
for item in filenames:
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    #degree_sequence = [log(x) for x in degree_sequence]
    
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    num_bins = len(degree_sequence)
    deg = [float(log10(x)) for x in deg]  

    if i == 0:
        max_t = (max(deg))
    axs[len(filenames)-1-i].hist(deg, len(deg), color=color[i], label = str(x[i]))
    axs[len(filenames)-1-i].set_xlim(0, max_t)
    i +=1
#plt.legend()
#plt.show()

plt.xlabel('log(degree)')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(name + '_hist')

from collections import Counter

i = 0
plt.figure(figsize=(6,6)) 
for item in filenames:
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    
    
    degreeCount = collections.Counter(degree_sequence)
    print(degreeCount)
    x1, y = zip(*degreeCount.items())                                                                                                                      

    x1 = [float(log10(x)) for x in x1]    
    y = [float(log10(x)) for x in y]                                                                                                          
    plt.scatter(x1, y, marker='.', s=100, color=color[i], label = str(x[i]))                                                                                                 
    i = i+1
plt.xlabel('log(degree)')   

                                                                                      
plt.ylabel('log(frequency)')   
plt.legend()
plt.savefig(name + '_scatter')
plt.show()
plt.close()
plt.cla()



plt.figure(figsize=(6, 6)) 
i = 0
for item in filenames:
    # Subset to the airline
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
    print(max(degree_sequence))

    degree_sequence = [float(log10(x)) for x in degree_sequence]
    print(max(degree_sequence))
    # Draw the density plot
    sns.distplot(degree_sequence, hist = False, kde = True,
                 kde_kws = {'shade': False,'linewidth': 3},color=color[i], label = str(x[i]))
    i += 1       
    
# Plot formatting
plt.legend()
#plt.title('Density Plot with Multiple Airlines')
plt.xlabel('LOG(degree)')
plt.ylabel('Density')
plt.savefig(name + '_density.png')

plt.show()