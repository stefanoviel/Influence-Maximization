import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filenames = ["scale_graphs/fb_politician.txt_TRUE-4.0.txt","graphs/fb_politician.txt"]
pp = []
mm = []
ma = []
kk = []


G = read_graph(filenames[1])
print(nx.info(G))
den = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
print("Density --> {0}".format(den))
my_degree_function = G.degree
mean = []
mean_degree = []
for item in G:
    mean.append(my_degree_function[item])
    mean_degree.append(float(1/my_degree_function[item]))

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

position = nx.spring_layout(G1)

nx.draw(G1, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
plt.show()

plt.cla()

#position = nx.spring_layout(G)

#nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
#plt.show()
#plt.cla()

import seaborn as sns
fig, axs = plt.subplots(ncols=2)
sns.distplot(mean, hist=True, kde=False, 
                color = 'darkblue', 
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 4},ax=axs[0])
#sns.distplot(mean, hist = False, kde = True,
#                kde_kws = {'shade': True, 'linewidth': 3},ax=axs[0])    
sns.distplot(mean_1, hist=True, kde=False, 
                color = 'darkblue', 
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 4},ax=axs[1])
#sns.distplot(mean_1, hist = False, kde = True,
#                kde_kws = {'shade': True, 'linewidth': 3},ax=axs[1])

kk = [mean, mean_1]
axs[0].set_title('Original Graph')  
axs[0].set_xlabel('In\out Degree')
axs[1].set_title('50% Graph')
axs[1].set_xlabel('In\out Degree')
plt.show()


x = ["Original","Scaled"]
i = 0
for item in kk:    
    # Draw the density plot
    print(item)
    sns.distplot(item, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3}, 
                label = str(x[i]))
    i +=1
plt.legend()
plt.title('Avg. Degree Nodes Distribution')
plt.xlabel('Avg Degree Nodes')
plt.ylabel('Density')
plt.show()



# G = read_graph("graphs/graph_SBM_small.txt")
# position = nx.spring_layout(G)
# nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
# plt.savefig("SBM", dpi=300)
# plt.cla()
