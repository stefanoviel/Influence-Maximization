import time
import numpy as np
import pandas as pd
from src.load import read_graph
from pymoo.indicators.hv import Hypervolume
from src_OLD.heuristics.SNInflMaxHeuristics_networkx import *


name = 'facebook_combined'
G = read_graph(f'scale_graphs/{name}_8.txt')
start_time = time.time()
S = CELF(int(G.number_of_nodes() * 0.025), G, 0, 100, 'WC')

#t = generalized_degree_discount(int(G.number_of_nodes() * 0.025), G, 0.05)

n_nodes = []
influence = []
nodes = []
pf = []
for item in S:
    n_nodes.append(item[0])
    influence.append(item[1])
    pf.append([-item[1], -(2.5 - item[0])])
    nodes.append(item[2])
df = pd.DataFrame()
df["n_nodes"] = n_nodes
df["influence"] = influence
df["nodes"] = nodes
df.to_csv(f'heuristics_experiment/{name}_8_WC.csv', index=False)




tot = 100 * 2.5 
pf = np.array(pf)
metric = Hypervolume(ref_point= np.array([0,0]),
                    norm_ref_point=False,
                    zero_to_one=False)

hv_original = metric.do(pf) /tot
print(hv_original)