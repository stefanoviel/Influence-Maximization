import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
import matplotlib.pyplot as plt
import  os
graphs = ["graphs/facebook_combined.txt","graphs/fb_org.txt","graphs/fb_politician.txt","graphs/pgp.txt","graphs/deezerEU.txt"]


for g in graphs:
    G = read_graph(g)
    position = nx.spring_layout(G)
    print(nx.info(G))
    nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
    g = os.path.basename(g)
    plt.savefig(g + '.png', dpi=1000)

