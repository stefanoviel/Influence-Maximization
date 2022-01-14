from pickle import TRUE
import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from math import log
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
plt.figure(figsize=(8, 8)) 
i = 0
for item in filenames:
    G = read_graph(item)   
    # print(np.mean(item), 100 - (np.mean(item)/real) *100)
    # sns.distplot(item, hist = False, kde = True,
    #              kde_kws = {'shade': False, 'linewidth': 3}, 
    #             label = str(x[i]))

    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.loglog(degrees, degree_freq,'go-', color=color[i], label = str(x[i]))
    i +=1
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('degree_log.png')

#plt.show()
plt.cla()
plt.close()


i=0
filenames = filenames[::-1]
x = x[::-1]
color = color[::-1]
plt.figure(figsize=(8, 8)) 
for item in filenames:
    G = read_graph(item)   
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    dmax = max(degree_sequence)
    plt.plot(degree_sequence, "b-", marker="o",color=color[i], label = str(x[i]))
    i +=1
plt.xlabel('Rank')
plt.ylabel('Degree')
plt.title('Degree Rank Plot')
plt.legend()
plt.savefig('degree_rank.png')
#plt.show()

i = 0
fig, axs = plt.subplots(4)
fig.suptitle('Degree Distribution')

import collections
for item in filenames:
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degree_sequence = [log(x) for x in degree_sequence]
    degree_sequence = sorted([x for x in degree_sequence], reverse=True)
    if i == 0:
        max_t = (max(degree_sequence))
    
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    num_bins = len(degree_sequence)
    axs[len(filenames)-1-i].bar(deg, cnt, width=0.80, color=color[i], label = str(x[i]))
    axs[len(filenames)-1-i].set_xlim(0, max_t)
    #axs[len(filenames)-1-i].legend()
    # sns.distplot(deg, hist = True, kde = True,
    #               kde_kws = {'shade': False, 'linewidth': 3}, 
    #              label = str(x[i]), ax=axs[len(filenames)-1-i])

    i +=1
#plt.legend()
#plt.show()
plt.tight_layout()
plt.savefig('degree_hist')

from collections import Counter

i = 0
plt.figure(figsize=(8, 8)) 
for item in filenames:
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    x1, y = zip(*degreeCount.items())                                                      
                                                                                                    
                                                                                                                                                                                                                                                        
    # prep axes                                                                                                                      
                                                                                                          
    plt.xscale('log')                                                                                                                
                                                                                                       
    plt.yscale('log')                                                                                                                
    plt.scatter(x1, y, marker='.', s=100, color=color[i], label = str(x[i]))                                                                                                 
    i = i+1
plt.xlabel('degree')   

                                                                                      
plt.ylabel('frequency')   
plt.legend()
plt.savefig('degree_scatter')

plt.close()
plt.cla()



plt.figure(figsize=(8, 8)) 
i = 0
for item in filenames:
    # Subset to the airline
    G = read_graph(item)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
    
    # Draw the density plot
    sns.distplot(degree_sequence, hist = False, kde = True,
                 kde_kws = {'shade': False,'linewidth': 3},color=color[i], label = str(x[i]))
    i += 1       
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'Airline')
plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.savefig('degree_prova.png')


'''
print(max(mean), np.mean(mean), np.std(mean))
G1 = read_graph(filenames[0])
print(nx.info(G1))
den1 = (2*G1.number_of_edges()) / (G1.number_of_nodes()*(G1.number_of_nodes()-1))
print("Density --> {0}".format(den1))
my_degree_function = G1.degree
mean_1 = []
mean_degree_1 = []
for item in G1:
    mean_1.append(my_degree_function[item])
    mean_degree_1.append(float(1/my_degree_function[item]))
print(max(mean_1), np.mean(mean_1), np.std(mean_1))
print(den1/den)

print(G1.number_of_nodes()/G.number_of_nodes())

#position = nx.spring_layout(G1)

#nx.draw(G1, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
##plt.show()
plt.savefig('')

#plt.cla()

#position = nx.spring_layout(G)

#nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
##plt.show()
plt.savefig('')
#plt.cla()
'''


'''
import seaborn as sns
fig, axs = plt.subplots(ncols=2)
sns.distplot(kk[3], hist=True, kde=False, 
                color = 'darkblue', 
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 4},ax=axs[0])
sns.distplot(kk[3], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 3},ax=axs[0])    
sns.distplot(kk[0], hist=True, kde=False, 
                color = 'darkblue', 
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 4},ax=axs[1])
sns.distplot(kk[0], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 3},ax=axs[1])



axs[0].set_title('Original Graph')  
axs[0].set_xlabel('In\out Degree')
axs[1].set_title('50% Graph')
axs[1].set_xlabel('In\out Degree')
#plt.show()
plt.savefig('')
'''