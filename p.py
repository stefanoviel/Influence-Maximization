import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#filenames = ["scale_graphs/graph_SBM_small_scale_5.txt","scale_graphs/graph_SBM_small_scale_4.txt","scale_graphs/graph_SBM_small_scale_3.txt","scale_graphs/graph_SBM_small_scale_2.txt","scale_graphs/graph_SBM_small_scale_1.5.txt","scale_graphs/graph_SBM_small_scale_1.33.txt","graphs/graph_SBM_small.txt"]
filenames = ["scale_graphs/facebook_combined.txt_example4.txt","graphs/facebook_combined.txt"]
#scale_graphs/graph_SBM_big.txt_example5.txt","graphs/graph_SBM_big.txt"]
pp = []
mm = []
ma = []
kk = []
# for filename in filenames:
#     G = read_graph(filename)
#     my_degree_function = G.degree
#     mean = []
#     mean_degree_1 = []
#     for item in G:
#         mean.append(my_degree_function[item])
#         mean_degree.append(float(1/my_degree_function[item]))

#     p = np.mean(mean_degree)
#     k = np.mean(mean)
#     mm.append(p)
#     print(p)
#     print(k)
#     print(p*k)
#     kk.append(mean)

#     print("-------")
#     sns.distplot(mean, hist=True, kde=True, 
#                  color = 'darkblue', 
#                 hist_kws={'edgecolor':'black'},
#                 kde_kws={'linewidth': 4})
#     sns.distplot(mean, hist = False, kde = True,
#                     kde_kws = {'shade': True, 'linewidth': 3})
    
#     plt.xlabel("In/Out - Degree")
#     plt.title('Original Graph')
#     plt.show()


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

print(np.mean(mean_degree), max(mean))
G1 = read_graph(filenames[0])
print(nx.info(G1))
den = (2*G1.number_of_edges()) / (G1.number_of_nodes()*(G1.number_of_nodes()-1))
print("Density --> {0}".format(den))
my_degree_function = G1.degree
mean_1 = []
mean_degree_1 = []
for item in G1:
    mean_1.append(my_degree_function[item])
    mean_degree_1.append(float(1/my_degree_function[item]))
print(np.mean(mean_degree_1), max(mean_1))

position = nx.spring_layout(G1)

nx.draw(G1, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
plt.show()
plt.cla()


import seaborn as sns
fig, axs = plt.subplots(ncols=2)
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

kk = [mean, mean_1]
axs[0].set_title('Original Graph')  
axs[0].set_xlabel('In\out Degree')
axs[1].set_title('50% Graph')
axs[1].set_xlabel('In\out Degree')
plt.show()


x = [50,100]
i = 0
for item in kk:    
    # Draw the density plot
    sns.distplot(item, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3}, 
                label = str(x[i]))
    i +=1
plt.legend()
plt.title('Avg. Degree Nodes Distribution')
plt.xlabel('Avg Degree Nodes')
plt.ylabel('Density')
plt.show()

fig = plt.figure()
ax = plt.axes()
#ax.plot(x, pp, color="black", label="mean influence", marker='o')
ax.plot(x, mm, color="red", marker="o",label="ideal mean")
plt.legend()


# G = read_graph("graphs/graph_SBM_small.txt")
# position = nx.spring_layout(G)
# nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
# plt.savefig("SBM", dpi=300)
# plt.cla()
