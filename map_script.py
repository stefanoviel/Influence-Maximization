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
def get_PF(myArray):
    myArray = myArray[myArray[:,0].argsort()]
    # Add first row to pareto_frontier
    pareto_frontier = myArray[0:1,:]
    # Test next row against the last row in pareto_frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x]
                for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier
    
scale_list = [32]
graph_list = ['fb-pages-artist']
model_list = ['IC']
for graph in graph_list:
    for model_ in model_list:
        for scale_number in scale_list:

            degree_measure = ['two-hop','page_rank', 'degree_centrality','katz_centrality', 'betweenness', 'closeness', 'eigenvector_centrality', 'core']
            degree_measure = ['page_rank']
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
            for j in range(10):
                df_scale_results = pd.read_csv("experiments/{0}_{1}-{2}/run-{3}.csv".format(graph,scale_number, model_,j+1),sep=",")
                df_scale_results = df_scale_results.sort_values(by="n_nodes", ascending=False)
                nodes = df_scale_results["nodes"].to_list()
                for jj in range(10):
                    #filename_original_results = "experiments/{0}_{1}/run-{3}.csv".format(graph,model_,jj+1)
                    filename_original_results = None
                    print(filename_original_results)
            
                    MAP_RESULTS = {}
                    for measure in degree_measure:
                        MAP_RESULTS[measure] = []
                    for measure in degree_measure:
                        print('Calculating ', measure , ' ......' )
                        scaled_table = pd.read_csv(f'map_files/{graph}-{scale_number}-{measure}.csv', sep=',')
                        original_table = pd.read_csv(f'map_files/{graph}-1-{measure}.csv', sep=',')

                        #print(scaled_table)
                        #print(original_table)
                        print(measure, '--> ok')
                        print('Start Mapping ...')
                        solution = []
                        NODES = nodes
                        for number, item in enumerate(nodes):
                            print('{0}/{1}'.format(number+1, len(nodes)))
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


                        print('End Mappping ....') 

                        from src.spread.monte_carlo import MonteCarlo_simulation


                        original_filename = "graphs/{0}.txt".format(graph)
                        p = 0.05
                        no_simulations = 25
                        model = model_
                        G = read_graph(original_filename)

                        nodes_ = []
                        comm = []
                        influence = []
                        n_nodes = []
                        print('Start Evaluation ....') 

                        pop = []
                        for idx, item in enumerate(solution):	
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
                        df_mapping.to_csv('prova-{0}_{1}_{2}-{3}.csv'.format(graph,model_,scale_number,measure), index=False)

                        exit(0)

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
                        MAP_RESULTS[measure].append(hv_MAP)

                    print(MAP_RESULTS)

                    measure = []
                    hv = []
                    gd = []
                    hv_MAP = []
                    for key,value in MAP_RESULTS.items():
                        measure.append(key)
                        hv.append(value[0])
                        gd.append(value[1])
                        hv_MAP.append(value[2])

                    df_final = pd.DataFrame()
                    df_final["measure"] = measure
                    df_final["Hyperarea"] = hv
                    df_final["GD"] = gd
                    df_final["HyperVolume"] = hv_MAP
                    df_final.to_csv('Mapping_Results_Script/{0}_{1}_{2}_MAPPING_{3}-{4}.csv'.format(graph,model_,scale_number,j+1,jj+1), index=False)