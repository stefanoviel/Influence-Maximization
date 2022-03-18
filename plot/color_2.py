import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
from src.load import read_graph
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

name = 'fb_org'
m = 'WC'



df = pd.read_csv('experiments/{0}_4-{1}/run-1.csv'.format(name,m), sep=",")
G = read_graph('scale_graphs/{0}_4.txt'.format(name))


print(nx.info(G))
color_map = []
df = df.sort_values(by="n_nodes", ascending=False)
list_item = df["nodes"]
for k in list_item:
    item = k
    break
item = item.replace('[',"")
item = item.replace(']',"")
item = item.replace(',',"")
n = item.split(" ")
x = [int(x) for x in n]
print(f'Number of elements in the biggest seed set : {len(x)}')
#color_map = ['red' if node in x else 'white' for node in G] 
POSITION = nx.spring_layout(G)
fig, ax1 = plt.subplots()

left, bottom, width, height = [0.2, 0.25, 0.6, 0.6]
ax2 = fig.add_axes([left, bottom, width, height])
nx.draw(G, POSITION,  edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, with_labels=False, ax=ax2)
ax1.axis('off')
#elif i == 1:
    #G = [G.subgraph(c).copy() for c in nx.connected_components(G)]
#    position = nx.spring_layout(G)
#    nx.draw_networkx(G, position,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)         

plt.savefig('net_images_png/fb_org_downscaled_network.png', format='png', dpi=1000)


seed = x
nodes = []
for node in G:
    if node not in x:
        nodes.append(node)
fig, ax1 = plt.subplots()

left, bottom, width, height = [0.2, 0.25, 0.6, 0.6]
ax2 = fig.add_axes([left, bottom, width, height])
nx.draw(G, POSITION,  nodelist=nodes,edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, with_labels=False, ax=ax2)
nx.draw(G, POSITION, nodelist=seed, edgecolors='black',node_color='red',arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, with_labels=False, ax=ax2)

ax1.axis('off')
plt.savefig('net_images_png/fb_org_downscaled_solutions.png', format='png', dpi=1000)


# df = pd.read_csv('{0}_{1}_4-page_rank.csv'.format(name,m), sep=",")
# df = df.sort_values(by="n_nodes", ascending=False)
# list_item = df["nodes"]
# for k in list_item:
#     item = k
#     break
# item = item.replace('[',"")
# item = item.replace(']',"")
# item = item.replace(',',"")
# n = item.split(" ")
# x = [int(x) for x in n]


# seed = x
# nodes = []
# for node in G:
#     if node not in x:
#         nodes.append(node)
# nx.draw(G, POSITION,  nodelist=nodes,edgecolors='black',node_color='white',arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, with_labels=False)
# nx.draw(G, POSITION, nodelist=seed, edgecolors='black',node_color='red',arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, with_labels=False)


# plt.savefig('net_images_png/fb_org_original_upscaled_solutions_v3.png', format='png', dpi=1000)
