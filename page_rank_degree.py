from pprint import pprint
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# local libraries
from src.load import read_graph

graphs = ['fb_politician','fb-pages-public-figure', 'facebook_combined', 'fb_org','pgp','deezerEU']
models = ['WC', 'IC']
for name in graphs:
    for model in models:
        G = read_graph(f'graphs/{name}.txt')
        scale_factors = [2,4,8]
        df_rank = pd.DataFrame()
        T = nx.pagerank(G, alpha = 0.85)


        data = pd.DataFrame()
        node = []
        centr = []

        for key,value in T.items():
            node.append(key)
            centr.append(value)


        
        
        rank = [x for x in range(1,len(node)+1)]
        data["node"] = node
        data["page_rank"] = centr



        data = data.sort_values(by='page_rank', ascending=False)
        data["overall_rank"] = rank

        community = pd.read_csv(f'comm_ground_truth/{name}.csv',sep=",")
        int_df = pd.merge(data, community, how ='inner', on =['node'])
        n_comm = len(set(int_df["comm"].to_list()))
        node = []
        rank_comm = []
        len_list = []
        for i in range(n_comm):
            list_comm = int_df[int_df["comm"] == i+1]
            len_list.append(len(list_comm))
            for i in range(len(list_comm)):
                node.append(list_comm["node"].iloc[i])
                rank_comm.append(i+1)

        data_comm = pd.DataFrame()
        data_comm["rank_comm"] = rank_comm
        data_comm["node"] = node


        int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
        print(int_df)
        for s in scale_factors:
            df = pd.read_csv(f'Mapping/{name}_{model}_{s}-page_rank.csv',sep=',')
            nodes = df['nodes'].to_list()
            final = []
            L = []
            for item in nodes:
                f = []
                item = item.replace("[","")
                item = item.replace("]","")
                item = item.replace(",","")
                nodes_split = item.split()  
                for node in nodes_split:
                    T = int_df[int_df['node'] == int(node)]
                    T = int(T.get(key = 'rank_comm'))
                    f.append(T)
                final.append(f)
            max_value = None
            for item in final:
                print(len(item), np.mean(item))
                L.append(int(len(item)))
                if max_value==None:
                    max_value = max(item)
                else:
                    if max_value < max(item):
                        max_value = max(item)
            fig = plt.figure(figsize =(14, 7))
            
            # Creating plot
            final = final[::-1]
            plt.boxplot(final)
            plt.ylabel('Page Rank')
            plt.xlabel('Seed Set')
            plt.xticks(rotation=90)
            # show plot
            print(len(L), len(final))
            plt.subplots_adjust(left=0.09,
            bottom=0.1, 
            right=0.95, 
            top=0.90, 
            wspace=0.05, 
            hspace=0.05)
            plt.title(f'{name} - Upscaled by {s} - {model} Model')
            plt.savefig(f'Page_rank_plot/{name}_{s}_{model}.png', format='png')
            #plt.show()