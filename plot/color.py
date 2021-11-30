import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
from src.load import read_graph
import matplotlib.pyplot as plt

df = pd.read_csv("facebook_1684_TRUE-4-k5-p0-WC-degree.csv", sep=",")
G = read_graph("scale_graphs/facebook_1684.txt_TRUE-4.txt")

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
    position = nx.spring_layout(G)

    nx.draw(G, position,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
    plt.show()
    plt.cla()
    break



