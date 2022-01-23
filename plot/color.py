import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
from src.load import read_graph
import matplotlib.pyplot as plt

df = pd.read_csv("deezerEU_False-8-k77-p0.05-IC-NEW_3_OBJ.csv", sep=",")
G = read_graph("scale_graphs/deezerEU.txt_False-8.txt")

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



