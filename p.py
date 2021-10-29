import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filenames = ["scale_graphs/facebook_combined_scale_5.txt","scale_graphs/facebook_combined_scale_4.txt","scale_graphs/facebook_combined_scale_3.txt","scale_graphs/facebook_combined_scale_2.txt","scale_graphs/facebook_combined_scale_1.5.txt","scale_graphs/facebook_combined_scale_1.33.txt","graphs/facebook_combined.txt"]
pp = []
mm = []
ma = []
kk = []
for filename in filenames:
    G = read_graph(filename)

    my_degree_function = G.degree
    mean = []
    mean_degree = []
    for item in G:
        mean.append(my_degree_function[item])
        mean_degree.append(float(1/my_degree_function[item]))

    p = np.mean(mean_degree)
    k = np.mean(mean)
    ma.append(max(mean))
    kk.append(mean)
    pp.append(p)
    mm.append(k)
    # print(p)
    # print(k)
    # print(max(mean_degree), min(mean_degree))

    print(k / G.number_of_nodes())
    print("-------")
    # sns.distplot(mean, hist=True, kde=True, 
    #              color = 'darkblue', 
    #             hist_kws={'edgecolor':'black'},
    #             kde_kws={'linewidth': 4})
    # sns.distplot(mean, hist = False, kde = True,
    #                 kde_kws = {'shade': True, 'linewidth': 3})
    plt.show()

x = [20,25,33,50,66,75,100]
i = 0
for item in kk:    
    # Draw the density plot
    sns.distplot(item, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                label = str(x[i]))
    i +=1
plt.legend()
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.plot(x, pp, color="black", label="mean influence", marker='o')
ax.plot(x, mm, color="red", marker="o",label="ideal mean")
plt.legend()
#plt.show()

print(pp)
print(mm)
print(ma)
# import leidenalg
# import igraph as ig
# filename = "graphs/ego-twitter.txt"
# G = read_graph(filename)
# print(nx.info(G))
# R = ig.Graph.from_networkx(G)
# part = leidenalg.find_partition(R, leidenalg.ModularityVertexPartition)
# check = list(part)
# print(len(check))
