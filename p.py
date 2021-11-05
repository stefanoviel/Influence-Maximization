import pandas as pd
import networkx as nx
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
x1 = [100,10]
x2 = [99,1000]

s = 1
for item in x1:
    s = s * item

print(s)
s = 1
for item in x2:
    s = s * item

print(s)
if (x1 > x2):
    print("True")
else:
    print("False")
exit(0)

filenames = ["scale_graphs/graph_SBM_small_scale_5.txt","scale_graphs/graph_SBM_small_scale_4.txt","scale_graphs/graph_SBM_small_scale_3.txt","scale_graphs/graph_SBM_small_scale_2.txt","scale_graphs/graph_SBM_small_scale_1.5.txt","scale_graphs/graph_SBM_small_scale_1.33.txt","graphs/graph_SBM_small.txt"]
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
    mm.append(p)
    # print(p)
    # print(k)
    # print(max(mean_degree), min(mean_degree))
    print(p)
    print(k)
    print(p*k)
    kk.append(mean)

    print("-------")
    sns.distplot(mean, hist=True, kde=True, 
                 color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4})
    sns.distplot(mean, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 3})
    plt.show()

x = [20,25,33,50,66,75,100]
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
