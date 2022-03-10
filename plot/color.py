import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '')
from src.load import read_graph
from src.load import read_graph
import matplotlib.pyplot as plt



graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
models = ['IC','WC']


for name in graphs:
    for m in models:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))

        list_graphs = ['graphs/{0}.txt'.format(name), 'scale_graphs/{0}_4.txt'.format(name), 'graphs/{0}.txt'.format(name)]
        list_results = ['experiments/{0}-{1}/run-1.csv'.format(name, m), 'experiments/{0}_4-{1}/run-1.csv'.format(name,m), '{0}_{1}_4-page_rank.csv'.format(name,m)]
        ax = [ax1, ax2, ax3]
        titles = ['Original', 'Scaled', 'Mapping']
        for i in range(len(list_graphs)):
            df = pd.read_csv(list_results[i], sep=",")
            G = read_graph(list_graphs[i])
            a = ax[i]

            print(list_results[i],list_graphs[i] )
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
            color_map = ['red' if node in x else 'white' for node in G] 
            if i == 0:
                POSITION = nx.spring_layout(G)
                nx.draw_networkx(G, POSITION,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)
            elif i == 1:
                #G = [G.subgraph(c).copy() for c in nx.connected_components(G)]
                position = nx.spring_layout(G)
                nx.draw_networkx(G, position,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)         
            else:
                nx.draw_networkx(G, POSITION,  edgecolors='black',node_color=color_map,arrowsize=1,node_size=20,linewidths=0.5, edge_color="#C0C0C0", width=0.5, ax=a, with_labels=False)
            
            a.set_title(titles[i])

        plt.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0, 
                    hspace=0.35)
        plt.savefig(f'net_images/graph_{name}-{m}.eps', format='eps')
        #plt.show()
        #plt.cla()



