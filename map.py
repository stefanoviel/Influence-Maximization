from src.load import read_graph
import pandas as pd
import numpy as np
import os
import time
import random
import logging
import networkx as nx
import numpy as np
import pandas as pd
# local libraries
from src.load import read_graph
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
def n_neighbor(G, id, n_hop):
    node = [id]
    node_visited = set()
    neighbors= []
    
    while n_hop !=0:
        neighbors= []
        for node_id in node:
            node_visited.add(node_id)
            neighbors +=  [id for id in G.neighbors(node_id) if id not in node_visited]
        node = neighbors
        n_hop -=1
        
        if len(node) == 0 :
            return neighbors 
        
    return list(set(neighbors))

scale_list = [8,4,2]
graph_list = ['deezerEU']
model_list = ['WC']
for graph in graph_list:
    for model_ in model_list:
        for scale_number in scale_list:

            degree_measure = ['two-hop','page_rank', 'degree_centrality','katz_centrality', 'betweenness', 'closeness', 'eigenvector_centrality', 'core']

            MAP_RESULTS = {}
            for measure in degree_measure:
                MAP_RESULTS[measure] = []
            filename = "scale_graphs/{0}_{1}.txt".format(graph, scale_number)
            scale_comm = "comm_ground_truth/{0}_{1}.csv".format(graph, scale_number)


            filename_original = "graphs/{0}.txt".format(graph)
            filename_original_comm = "comm_ground_truth/{0}.csv".format(graph)





            G = read_graph(filename)
            G1 = read_graph(filename_original)

            scale_factor = int(G1.number_of_nodes() / G.number_of_nodes())

            scale_original = G1.number_of_nodes() / G.number_of_nodes()


            print('Scale Factor',scale_factor, scale_original)
            df_scale_results = pd.read_csv("experiments/{0}_{1}-{2}/run-1.csv".format(graph,scale_number, model_),sep=",")
            df_scale_results = df_scale_results.sort_values(by="n_nodes", ascending=False)
            nodes = df_scale_results["nodes"].to_list()

            filename_original_results = "experiments/{0}-{1}/run-1.csv".format(graph, model_)

            def get_table(graph_name, comm_name, measure):
                G = read_graph(filename=graph_name)

                from networkx.algorithms import degree_centrality, closeness_centrality, core_number, betweenness_centrality
                from networkx.algorithms import katz_centrality, katz_centrality_numpy, eigenvector_centrality_numpy, current_flow_betweenness_centrality
                if measure == 'page_rank':
                    T = nx.pagerank(G, alpha = 0.85)
                elif measure == 'degree_centrality':
                    T = degree_centrality(G)
                elif measure == 'katz_centrality':
                    T = katz_centrality_numpy(G)
                elif measure == 'betweenness':
                    T = betweenness_centrality(G)
                elif measure == 'eigenvector_centrality':
                    T = eigenvector_centrality_numpy(G)
                elif measure == 'closeness':
                    T = closeness_centrality(G)
                elif measure == 'core':
                    G.remove_edges_from(nx.selfloop_edges(G))
                    T = core_number(G)
                elif measure == 'two-hop':
                    T = {}
                    for node in G:
                        T[node] = len(n_neighbor(G,node,2))


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

                community = pd.read_csv(comm_name,sep=",")

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
                
                return int_df



            for measure in degree_measure:
                print('Calculating ', measure , ' ......' )
                scaled_table = get_table(filename, scale_comm, measure)
                original_table = get_table(filename_original, filename_original_comm, measure)

                print(measure, '--> ok')

                solution = []
                NODES = nodes
                for item in nodes:
                    item = item.replace("[","")
                    item = item.replace("]","")
                    item = item.replace(",","")
                    nodes_split = item.split()  

                    #print("---------")
                    #print(len(nodes_split),len(nodes_split)*scale_factor , scale_factor)
                    N = []
                    for node in nodes_split:
                        node = int(node)
                        t = scaled_table.loc[scaled_table["node"] == node]
                        if len(t) > 0:
                            r = original_table[original_table["comm"] == int(t.comm)]
                            n = r["node"].to_list()
                            l = r["page_rank"].to_list()
                            #print(n)
                            s = 0
                            ii = 0
                            #print('l3n', len(l))
                            #if len(l) == 0:
                                #print(r)
                                #print(n)
                            while ii < int(scale_factor):
                                myArray = np.array(l)                
                                pos = (np.abs(myArray-float(t.rank_comm))).argmin()

                                #pos = (np.abs(myArray-float(t.page_rank))).argmin()
                                if n[pos] in N:# and len(n) > 0:
                                    l = np.delete(myArray, pos)
                                    n = np.delete(n, pos)
                                else:
                                    N.append(n[pos])
                                    ii +=1
                                    s +=1
                                    l = np.delete(myArray, pos)
                                    n = np.delete(n, pos)
                                
                                if len(l) == 0:
                                    break
                            #print(s)
                        # try:
                        #     N.append(int(r1.node))
                        # except:
                        #     print(r["rank_comm"])
                        #     print(int(t.rank_comm))

                    if len(N) != int(len(nodes_split) * (scale_original)):
                        #print('probelm here', int(len(nodes_split) * (scale_original)) - len(N))
                        k = int(len(nodes_split) * (scale_original)) - len(N)
                        new_node = []
                        overall_rank = []
                        df_split = pd.DataFrame()
                        df_split["node"] = [int(x) for x in nodes_split]
                        #print(scaled_table)
                        #print(df_split)
                        df_new = pd.merge(scaled_table, df_split, on='node')

                        n = df_new["node"].to_list()[:k]

                        
                        for node in n:
                            node = int(node)
                            t = scaled_table.loc[scaled_table["node"] == node]
                            if len(t) > 0:
                                r = original_table[original_table["comm"] == int(t.comm)]
                                n = r["node"].to_list()
                                l = r["page_rank"].to_list()
                                #print(n)
                                s = 0
                                ii = 0
                                #print('l3n', len(l))
                                #if len(l) == 0:
                                    #print(r)
                                    #print(n)
                                while ii < 1:
                                    myArray = np.array(l)                
                                    pos = (np.abs(myArray-float(t.rank_comm))).argmin()

                                    #pos = (np.abs(myArray-float(t.page_rank))).argmin()
                                    if n[pos] in N:# and len(n) > 0:
                                        l = np.delete(myArray, pos)
                                        n = np.delete(n, pos)
                                    else:
                                        N.append(n[pos])
                                        ii +=1
                                        s +=1
                                        l = np.delete(myArray, pos)
                                        n = np.delete(n, pos)
                                    
                                    if len(l) == 0:
                                        break
                    solution.append(N)


                    

                from src.spread.monte_carlo import MonteCarlo_simulation


                original_filename = "graphs/{0}.txt".format(graph)
                p = 0.05
                no_simulations = 100
                model = model_
                G = read_graph(original_filename)

                nodes_ = []
                comm = []
                influence = []
                n_nodes = []

                pop = []
                for idx, item in enumerate(solution):
                    # l = len(item)
                    # item = []
                    # while len(item) < l:
                    #     import random
                    #     t = random.randrange(0, G.number_of_nodes())
                    #     if t not in item:
                    #         item.append(t)	
                    A = set(item)
                    NODES[idx] = NODES[idx].replace("[","")
                    NODES[idx] = NODES[idx].replace("]","")
                    NODES[idx] = NODES[idx].replace(",","")
                    nodes_split = NODES[idx].split() 
                    print(measure, ' ', idx,'/',len(solution))
                    #print(len(item), len(A), len(nodes_split), int(len(nodes_split)* (scale_original)))
                    spread  = MonteCarlo_simulation(G, A, p, no_simulations, model,  [], random_generator=None)
                    #print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
                    influence.append(((spread[0] / G.number_of_nodes())* 100))
                    nodes_.append(((len(A) / G.number_of_nodes())* 100))
                    n_nodes.append(list(A))
                    T = [((spread[0] / G.number_of_nodes())* 100), -((len(A) / G.number_of_nodes())* 100)]
                    
                    pop.append(T)

                df_mapping = pd.DataFrame()
                df_mapping["n_nodes"] = nodes_
                df_mapping["influence"] = influence
                df_mapping["nodes"] = n_nodes
                df_mapping.to_csv('{0}_{1}_{2}-{3}.csv'.format(graph,model_,scale_number,measure), index=False)



            #--------

                x_mapping =  df_mapping["n_nodes"].to_list()
                z_mapping = df_mapping["influence"].to_list()
                A = []
                A_G = []
                for i in range(len(x_mapping)):
                    A.append([-z_mapping[i],- (2.5 - x_mapping[i])])
                    A_G.append([z_mapping[i], x_mapping[i]])

                df = pd.read_csv(filename_original_results, sep=",")
                x_original = df["n_nodes"].to_list()
                z_original = df["influence"].to_list()


                pf = []
                pf_G = []
                
                for i in range(len(x_original)):
                    pf.append([-z_original[i],-(2.5 - x_original[i])])
                    pf_G.append([z_original[i], x_original[i]])


                from pymoo.factory import get_performance_indicator
                pf = np.array(pf)
                A = np.array(A)
                pf_G = np.array(pf_G)
                A_G = np.array(A_G)

                gd = get_performance_indicator("gd", pf_G)
                gd_distance = gd.do(A_G)


                tot = 100 * 2.5 
                from pymoo.indicators.hv import Hypervolume

                metric = Hypervolume(ref_point= np.array([0,0]),
                                    norm_ref_point=False,
                                    zero_to_one=False)

                hv_original = metric.do(pf) /tot

                hv_MAP = metric.do(A) / tot


                print(hv_MAP/hv_original, gd_distance)
                MAP_RESULTS[measure].append(hv_MAP/hv_original)
                MAP_RESULTS[measure].append(gd_distance)

            print(MAP_RESULTS)

            measure = []
            hv = []
            gd = []
            for key,value in MAP_RESULTS.items():
                measure.append(key)
                hv.append(value[0])
                gd.append(value[1])

            df_final = pd.DataFrame()
            df_final["measure"] = measure
            df_final["Hyperarea"] = hv
            df_final["GD"] = gd
            df_final.to_csv('{0}_{1}_{2}_MAPPING.csv'.format(graph,model_,scale_number), index=False)
