import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
import matplotlib.pyplot as plt
import  os
graphs = ["scale_graphs/CA-GrQc_2.txt"]


for g in graphs:
    G = read_graph(g)
    print(nx.info(G))
    position = nx.spring_layout(G)
    nx.draw(G, position,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=1, edge_color="#C0C0C0", width=0.5)
    plt.show()
    #g = os.path.basename(g)
    #plt.savefig(g + '.png', dpi=1000)

