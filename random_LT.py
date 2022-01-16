from re import M
from src.load import read_graph
from src.spread.monte_carlo import MonteCarlo_simulation

G = read_graph('graphs/facebook_combined.txt')

N = 1
MAX = 100
no_simulations = 100

m_values = G.number_of_nodes()
seed_sets = []

for i in range(N):
    import random
    seed_sets.append(random.sample(range(1, m_values), 100))

import pandas as pd
communities =[]
df = pd.read_csv('comm_ground_truth/facebook_combined.csv',sep=",")
print(df)
groups = df.groupby('comm')['node'].apply(list)
print(groups)
df = groups.reset_index(name='nodes')
communities = df["nodes"].to_list()

p = 0
model = ["LT2","LT"]
import time



for m in model:
    start = time.time()
    for id, item in enumerate(seed_sets):
        A = set(item)
        print(len(A))
        influence,_, t, comm = MonteCarlo_simulation(G, A, p, no_simulations, m, communities, random_generator=None)
        print(influence)

    exec_time = time.time() - start   
    print(exec_time)   
