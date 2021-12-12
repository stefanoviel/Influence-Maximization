import networkx as nx
from networkx.algorithms.centrality.degree_alg import degree_centrality
from src.load import read_graph
import pandas as pd
filename = "scale_graphs/graph_SBM_small.txt_TRUE-4.txt"

G = read_graph(filename=filename)


T = degree_centrality(G)
#T = nx.pagerank(G, alpha = 0.8)
df = pd.read_csv("graph_SBM_small_TRUE-4-k25-p0.05-IC-degree-pop.csv",sep=",")

nodes = df["nodes"]


data = pd.DataFrame()
node = []
centr = []

for key,value in T.items():
    node.append(key)
    centr.append(value)

rank = [x for x in range(1,len(node)+1)]
data["node"] = node
data["centrality"] = centr

data = data.sort_values(by='centrality', ascending=False)
data["rank"] = rank
data.to_csv("prova.csv", index=False)

print(data)
community = pd.read_csv("comm_ground_truth/graph_SBM_small_4.csv",sep=",")
print(community)

int_df = pd.merge(data, community, how ='inner', on =['node'])
print(int_df)

n_comm = len(set(int_df["comm"].to_list()))
print(n_comm)
node = []
rank_comm = []
for i in range(n_comm):
    list_comm = int_df[int_df["comm"] == i+1]
    print(list_comm)
    for i in range(len(list_comm)):
        node.append(list_comm["node"].iloc[i])
        rank_comm.append(i+1)

data_comm = pd.DataFrame()
data_comm["rank_comm"] = rank_comm
data_comm["node"] = node


data_comm.to_csv("prova1.csv", index=False)
int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
print(int_df)
solution_nodes = []
soluition_comm = []
for item in nodes:
    item = item.replace("[","")
    item = item.replace("]","")
    item = item.replace(",","")
    nodes = item.split()    
    print("---------")
    A = []
    C = []
    R = []
    for node in nodes:
        node = int(node)
        t = int_df.loc[int_df["node"] == node]
        print(int(node), int(t.comm), int(t.rank_comm))
        C.append((t.comm))
        R.append((t.rank_comm))



    #print(A)
    #print(C)
    from src.spread.monte_carlo import MonteCarlo_simulation


    original_filename = "graphs/graph_SBM_small.txt"
    p = 0.05
    no_simulations = 100
    model = "IC"
    G = read_graph(original_filename)

    df = pd.read_csv("comm_ground_truth/graph_SBM_small.csv",sep=",")
    groups = df.groupby('comm')['node'].apply(list)
    df = groups.reset_index(name='nodes')
    communities_original = df["nodes"].to_list()


    spread  = MonteCarlo_simulation(G, A, p, no_simulations, model, communities_original, random_generator=None)
    print(spread)


