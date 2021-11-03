import networkx as nx
from networkx.algorithms.centrality.degree_alg import degree_centrality
from src.load import read_graph
import pandas as pd
filename = "scale_graphs/facebook_combined_scale_4.txt"

G = read_graph(filename=filename)


T = degree_centrality(G)
#T = nx.pagerank(G, alpha = 0.8)
df = pd.read_csv("/Users/elia/Desktop/Influence-Maximization/facebook_combined_scale_4-k25-p0.2-IC.csv.csv",sep=",")

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
data.to_csv("prova.csv")

print(data)
community = pd.read_csv("comm_ground_truth/facebook_combined_4.csv",sep=",")
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

int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
print(int_df)
#groups = community.groupby('comm')['node'].apply(list)
#communities = groups.reset_index(name='nodes')
#communities = communities["nodes"].to_list()
for item in nodes:

    item = item.replace("[","")
    item = item.replace("]","")
    item = item.replace(",","")
    nodes = item.split()    
    print("---------")
    A = []
    C = []
    for node in nodes:
        node = int(node)
        t = int_df.loc[int_df["node"] == node]
        print(t)

        #A.append(node)
        #C.append(int(c["comm"]))




print(A)
print(C)
from src.spread.monte_carlo import MonteCarlo_simulation


original_filename = "graphs/facebook_combined.txt"
p = 0.05
no_simulations = 100
model = "IC"


df = pd.read_csv("/Users/elia/Desktop/Influence-Maximization/comm_ground_truth/facebook_combined.csv",sep=",")
groups = df.groupby('comm')['node'].apply(list)
df = groups.reset_index(name='nodes')
communities_original = df["nodes"].to_list()


spread , time = MonteCarlo_simulation(G, A, p, no_simulations, model, communities_original, random_generator=None)
print(spread)

