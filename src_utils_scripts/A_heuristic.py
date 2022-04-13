import time
import numpy as np
import pandas as pd
from src.load import read_graph
from src_OLD.heuristics.SNInflMaxHeuristics_networkx import *
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation


G = read_graph('graphs/soc-brightkite.txt')


model = 'WC'

heuristic = 'CELF'

if heuristic == 'low_distance_nodes':
   H = low_distance_nodes(int(G.number_of_nodes()*0.025),G)
   s = []
   for item in H:
      s.append(item[1])
   influence = []
   nodes_ = []
   n_nodes = []
   for i in range(len(s)):
      A = s[:i+1]
      print(i+1,'/',len(s))
      spread  = MonteCarlo_simulation(G, A, 0.05, 100, model,  [], random_generator=None)
      print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
      influence.append(((spread[0] / G.number_of_nodes())* 100))
      n_nodes.append(((len(A) / G.number_of_nodes())* 100))
      nodes_.append(list(A))
      df = pd.DataFrame()
      df["n_nodes"] = n_nodes
      df["influence"] = influence
      df["nodes"] = nodes_
      df.to_csv(f'soc-brightkite_{heuristic}_runtime_{model}.csv', index=False)

   df = pd.DataFrame()
   df["n_nodes"] = n_nodes
   df["influence"] = influence
   df["nodes"] = nodes_


   df.to_csv(f'soc-brightkite_FINAL_{heuristic}_{model}.csv', index=False)

elif heuristic == 'high_degree_nodes':
   H = high_degree_nodes(int(G.number_of_nodes()*0.025),G)
   s = []
   for item in H:
      s.append(item[1])
   influence = []
   nodes_ = []
   n_nodes = []
   for i in range(len(s)):
      A = s[:i+1]
      print(i+1,'/',len(s))
      spread  = MonteCarlo_simulation(G, A, 0.05, 100, model,  [], random_generator=None)
      print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
      influence.append(((spread[0] / G.number_of_nodes())* 100))
      n_nodes.append(((len(A) / G.number_of_nodes())* 100))
      nodes_.append(list(A))
      df = pd.DataFrame()
      df["n_nodes"] = n_nodes
      df["influence"] = influence
      df["nodes"] = nodes_
      df.to_csv(f'soc-brightkite_{heuristic}_runtime_{model}.csv', index=False)

   df = pd.DataFrame()
   df["n_nodes"] = n_nodes
   df["influence"] = influence
   df["nodes"] = nodes_
   df.to_csv(f'soc-brightkite_FINAL_{heuristic}_{model}.csv', index=False)

elif heuristic == 'single_discount_high_degree_nodes':
   s = single_discount_high_degree_nodes(int(G.number_of_nodes()*0.025),G)
   influence = []
   nodes_ = []
   n_nodes = []
   for i in range(len(s)):
      A = s[:i+1]
      print(i+1,'/',len(s))
      spread  = MonteCarlo_simulation(G, A, 0.05, 100, model,  [], random_generator=None)
      print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
      influence.append(((spread[0] / G.number_of_nodes())* 100))
      n_nodes.append(((len(A) / G.number_of_nodes())* 100))
      nodes_.append(list(A))
      df = pd.DataFrame()
      df["n_nodes"] = n_nodes
      df["influence"] = influence
      df["nodes"] = nodes_


      df.to_csv(f'soc-brightkite_{heuristic}_runtime_{model}.csv', index=False)

   df = pd.DataFrame()
   df["n_nodes"] = n_nodes
   df["influence"] = influence
   df["nodes"] = nodes_


   df.to_csv(f'soc-brightkite_FINAL_{heuristic}_{model}.csv', index=False)

elif heuristic == 'CELF':

   RES = CELF(int(G.number_of_nodes()*0.025),G,0.5, 100, model)
   S = RES[0]
   influence = []
   nodes_ = []
   n_nodes = []
   for item in S:
      influence.append(item[1])
      nodes_.append(item[2])
      n_nodes.append(item[0])
   df = pd.DataFrame()
   df["n_nodes"] = n_nodes
   df["influence"] = influence
   df["nodes"] = nodes_
   df.to_csv(f'soc-brightkite_FINAL_{heuristic}_{model}.csv', index=False)