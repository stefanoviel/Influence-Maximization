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
from src.spread.monte_carlo import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
from new_ea import moea_influence_maximization




filename = "scale_graphs/pgp_8.txt"
scale_comm = "comm_ground_truth/pgp_8.csv"


filename_original = "graphs/pgp.txt"
filename_original_comm = "comm_ground_truth/pgp.csv"


G = read_graph(filename)
G1 = read_graph(filename_original)
scale_factor = G1.number_of_nodes() / G.number_of_nodes()

df = pd.read_csv("experiments/pgp_8-WC/run-1.csv",sep=",")

nodes = df["nodes"].to_list()

def normalize_list(list_normal):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        list_normal[i] = (list_normal[i] - min_value) / (max_value - min_value)
    return list_normal

def get_table(graph_name, comm_name,w):
    G = read_graph(filename=graph_name)

    from networkx.algorithms import degree_centrality, closeness_centrality
    T = degree_centrality(G)
    #T = nx.eigenvector_centrality(G)
    #T = closeness_centrality(G)
    T = nx.pagerank(G, alpha = 0.85)
    data = pd.DataFrame()
    node = []
    centr = []
    z = 0
    zz = 0
    for key,value in T.items():
        node.append(key)
        #centr.append(value / w)
        centr.append(value)

        #zz += F[key]
        z += value
    print(z)

    #centr = normalize_list(centr)
    
    
    rank = [x for x in range(1,len(node)+1)]
    data["node"] = node
    data["page_rank"] = centr



    data = data.sort_values(by='page_rank', ascending=False)
    data["rank"] = rank
    data.to_csv("prova.csv", index=False)

    community = pd.read_csv(comm_name,sep=",")
    #print(community)

    int_df = pd.merge(data, community, how ='inner', on =['node'])
    n_comm = len(set(int_df["comm"].to_list()))
    #print(n_comm)
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


    #data_comm.to_csv("comm_ground_truth/graph_SBM_big.csv", index=False)
    int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
    #print(int_df)
    
    return int_df
    #return len_list


print('Scale')
scaled_table = get_table(filename, scale_comm, scale_factor)
scaled_table.to_csv('scale.csv',index=False)
print('Original')
original_table = get_table(filename_original, filename_original_comm,1)
original_table.to_csv('original.csv', index=False)
#for i in range(len(scaled_table)):
#    print('Original {0}, Scaled {1}, Expected {2}'.format(original_table[i], scaled_table[i],int(original_table[i] / int(8))))

new = pd.merge(scaled_table,original_table, how='inner',on=['rank_comm', 'comm'])
print(new)

n_scale = new["node_x"].to_list()
n_origin = new["node_y"].to_list()


solution = []
NODES = nodes
for item in nodes:
    item = item.replace("[","")
    item = item.replace("]","")
    item = item.replace(",","")
    nodes_split = item.split()  

    #print("---------")
    #print(len(nodes_split),len(nodes_split)*scale_factor )
    N = []
    for node in nodes_split:
        #print(k)
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
                
                if len(n) == 0:
                    print('shit')
                if len(l) == 0:
                    print('shit')
                    break
            print(s)
        # try:
        #     N.append(int(r1.node))
        # except:
        #     print(r["rank_comm"])
        #     print(int(t.rank_comm))
    solution.append(N)
    print(len(N), len(set(N)))
    if len(N) != len(nodes_split) * int(scale_factor):
        print('cazzo')
        #exit(0)

#print(solution, len(solution), len(nodes))
        #n = r["node"]
        #print(n[pos])

        #C.append((t.comm))
        #R.append((t.rank_comm))
    
    #print(A)
    #print(C)
    

from src.spread.monte_carlo import MonteCarlo_simulation


original_filename = "graphs/pgp.txt"
p = 0.05
no_simulations = 100
model = "WC"
G = read_graph(original_filename)

df = pd.read_csv("comm_ground_truth/pgp.csv",sep=",")
groups = df.groupby('comm')['node'].apply(list)
df = groups.reset_index(name='nodes')
communities_original = df["nodes"].to_list()
print(len(communities_original))
nodes_ = []
comm = []
influence = []

# my_degree_function = G.degree
# mean = []
# for item in G:
#     mean.append(my_degree_function[item])

# args = {}
# args["filter_best_spread_nodes"] = True
# args["search_space_size_max"] = 1e11
# args["search_space_size_min"] = 1e9
# args["k"] = int(G.number_of_nodes() * 0.025)
# random_seed = 10
# prng = random.Random(random_seed)


# args["model"] = model
# args["p"] = p
# args["min_degree"] = np.mean(mean) + 1

# nn = filter_nodes(G, args)

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
    print(len(item), len(A), len(nodes_split), len(nodes_split)* int(scale_factor))
    try:
        spread  = MonteCarlo_simulation(G, A, p, no_simulations, model, communities_original, random_generator=None)
        print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
        influence.append(((spread[0] / G.number_of_nodes())* 100))
        nodes_.append(((len(A) / G.number_of_nodes())* 100))
        comm.append(spread[2])
        #T = [((spread[0] / G.number_of_nodes())* 100), -((len(A) / G.number_of_nodes())* 100),spread[2]]
        T = [((spread[0] / G.number_of_nodes())* 100), -((len(A) / G.number_of_nodes())* 100)]

        pop.append(T)
    except:
        pass
df = pd.DataFrame()
df["n_nodes"] = nodes_
df["influence"] = influence
df["communities"] = comm

df.to_csv('map_degree_comm.csv', index=False)
#print(len(df))


exit(0)
