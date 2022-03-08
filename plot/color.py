import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
from src.load import read_graph
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))

list_graphs = ['graphs/fb-pages-public-figure.txt', 'scale_graphs/fb-pages-public-figure_4.txt', 'graphs/fb-pages-public-figure.txt']
list_results = ['experiments/fb-pages-public-figure-IC/run-1.csv', 'experiments/fb-pages-public-figure_4-IC/run-1.csv', 'fb-pages-public-figure_IC_8-page_rank.csv']
ax = [ax1, ax2, ax3]
k = ['Original', 'Scaled', 'Mapping']
for i in range(len(list_graphs)):
    df = pd.read_csv(list_results[i], sep=",")
    G = read_graph(list_graphs[i])
    a = ax[i]


    print(nx.info(G))
    color_map = []

    df = df.sort_values(by="n_nodes", ascending=False)
    nodes = df["nodes"]

    for item in nodes:
        item = item.replace('[',"")
        item = item.replace(']',"")
        item = item.replace(',',"")
        print(item)
        n = item.split(" ")
        x = [int(x) for x in n]
        print(len(x))
        color_map = ['red' if node in x else 'white' for node in G] 
        if i == 0:
            POSITION = nx.spring_layout(G)
            nx.draw_networkx(G, POSITION,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)
        elif i == 1:
            position = nx.spring_layout(G)
            nx.draw_networkx(G, position,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)         
        else:
            nx.draw_networkx(G, POSITION,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)
           
        a.set_title(k[i])
        break

plt.subplots_adjust(left=0.01,
            bottom=0.01, 
            right=0.99, 
            top=0.95, 
            wspace=0, 
            hspace=0.35)
plt.savefig('graph.eps', format='eps')
plt.show()
        #plt.cla()



