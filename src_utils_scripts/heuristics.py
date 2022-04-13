import sys
import argparse
import pandas as pd
sys.path.insert(0, '')
from src.load import read_graph
from src_OLD.heuristics.SNInflMaxHeuristics_networkx import *
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation

def read_arguments():
   parser = argparse.ArgumentParser(
      description='Heuristic algorithm computation.'
   )
   # Problem setup.
   parser.add_argument('--graph', default='facebook_combined',
                     choices=['facebook_combined', 'fb_politician',
                              'deezerEU', 'fb_org', 'fb-pages-public-figuree',
                              'pgp', 'soc-gemsec', 'soc-brightkite'],
                     help='Graph name')

   # Upscaling Parameters
   parser.add_argument('--s', type=int, default=4,
                     help='Scaling factor')  
   parser.add_argument('--heuristic', default='CELF',
                     choices=['low_distance_nodes','high_degree_nodes', 'single_discount_high_degree_nodes',
                              'CELF'])
   parser.add_argument('--p', type=float, default=args["p"],
                  help='Probability of influence spread in the IC model.')
   parser.add_argument('--no_simulations', type=int, default=100,
                  help='Number of simulations for spread calculation'
                        ' when the Monte Carlo fitness function is used.')
   parser.add_argument('--model', default="WC", choices=['IC', 'WC'],
                  help='Influence propagation model.')
   parser.add_argument('--k', default=2.5, type=int,
                  help='Percentage of nodes as seed sets')
   
   args = parser.parse_args()
   args = vars(args)

   return args

def get_propagation(G,s,args):
   influence = []
   nodes_ = []
   n_nodes = []
   for i in range(len(s)):
      A = s[:i+1]
      print(i+1,'/',len(s))
      spread  = MonteCarlo_simulation(G, A, args["p"], 100, model, random_generator=None)
      print(((spread[0] / G.number_of_nodes())* 100), spread[2], ((len(A) / G.number_of_nodes())* 100))
      influence.append(((spread[0] / G.number_of_nodes())* 100))
      n_nodes.append(((len(A) / G.number_of_nodes())* 100))
      nodes_.append(list(A))
   df = pd.DataFrame()
   df["n_nodes"] = n_nodes
   df["influence"] = influence
   df["nodes"] = nodes_
   df.to_csv('experiments_heuristic/{0}_{1}_{2}.csv'.format(args["graph"],args["heuristic"],args["model"]), index=False)

if __name__ == '__main__':
   args = read_arguments() 

   G = read_graph('graphs/{0}.txt'.format(args["graph"]))
   heuristic = args["heuristic"]
   model = args["model"]

   if heuristic == 'low_distance_nodes':
      H = low_distance_nodes(int(G.number_of_nodes()*(args["k"]*100)),G)
      s = []
      for item in H:
         s.append(item[1])
      get_propagation(G,s,args)

   elif heuristic == 'high_degree_nodes':
      H = high_degree_nodes(int(G.number_of_nodes()*(args["k"]*100)),G)
      s = []
      for item in H:
         s.append(item[1])
      get_propagation(G,s,args)


   elif heuristic == 'single_discount_high_degree_nodes':
      s = single_discount_high_degree_nodes(int(G.number_of_nodes()*(args["k"]*100)),G)
      get_propagation(G,s,args)


   elif heuristic == 'CELF':

      RES = CELF(int(G.number_of_nodes()*(args["k"]*100)),G,args["p"], args["no_simulations"], model)
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
      df.to_csv('{0}_{1}_{2}.csv'.format(args["graph"],args["heuristc"], args["model"]), index=False)