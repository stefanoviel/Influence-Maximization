import time
import numpy as np
import pandas as pd
from src.load import read_graph
from pymoo.indicators.hv import Hypervolume
from src_OLD.heuristics.SNInflMaxHeuristics_networkx import *
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation


G = read_graph('graphs/fb-pages-artist.txt')
start_time = time.time()
#S = CELF(100, G, 0.05, 1, 'IC')


H = high_degree_nodes(int(G.number_of_nodes()*0.025),G)

s = []
for item in H:
    print(type(item))
    s.append(item[1])
print(s)
#print(S)
# print('Running ....')
# s = single_discount_high_degree_nodes(int(G.number_of_nodes()*0.025),G)

influence = []
nodes_ = []
n_nodes = []
for i in range(len(s)):

    A = s[:i+1]
    print(i+1,'/',len(s))
    spread  = MonteCarlo_simulation(G, A, 0, 10, 'WC',  [], random_generator=None)
    print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
    influence.append(((spread[0] / G.number_of_nodes())* 100))
    n_nodes.append(((len(A) / G.number_of_nodes())* 100))
    nodes_.append(list(A))

#S = CELF(int(G.number_of_nodes()*0.025),G,0, 10, 'WC')
#S = general_greedy((int(G.number_of_nodes()*0.025)),G,0, 1, 'WC')

#for item in S:
#    influence.append(item[1])
#    nodes_.append(item[2])
#    n_nodes.append(item[0])

df = pd.DataFrame()
df["n_nodes"] = n_nodes
df["influence"] = influence
df["nodes"] = nodes_

df.to_csv(f'heuristics_experiment/high_degree_nodes_WC_10.csv', index=False)


exit(0)

tot = 100 * 2.5 
pf = np.array(pf)
metric = Hypervolume(ref_point= np.array([0,0]),
                    norm_ref_point=False,
                    zero_to_one=False)

hv_original = metric.do(pf) /tot
print(hv_original)