import leidenalg
import igraph as ig
from src.load import read_graph
import networkx as nx

filename = "/Users/elia/Downloads/soc-gplus.txt"
G = read_graph(filename)
print(nx.info(G)) 

G = ig.Graph.from_networkx(G)
part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition);
print(type(part))
part = list(part)
print(part)

c = 0
cc = 0
for item in part:
    cc = cc + len(item)
    c = c+1

print(c,cc)