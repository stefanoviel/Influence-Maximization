import os
import time
import random
import logging
import networkx as nx
from functools import partial
import numpy as np
import pandas as pd
# local libraries
from src.load import read_graph
from src.spread.monte_carlo import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as monte_carlo_max_hop

from new_ea import moea_influence_maximization
from src.nodes_filtering.select_best_spread_nodes import filter_best_nodes as filter_best_spread_nodes
from src.nodes_filtering.select_min_degree_nodes import filter_best_nodes as filter_min_degree_nodes
from src.utils import inverse_ncr, community_detection
from src.smart_initialization import max_centrality_individual, Community_initialization, degree_random

def create_initial_population(G, args, prng=None, nodes=None):
    """
	Smart initialization techniques.
	"""
    # smart initialization
    initial_population = None

    initial_population = degree_random(args["k"], G,
                                        int(args["population_size"] * args["smart_initialization_percentage"]),
                                        prng, nodes=nodes)

    return initial_population
def filter_nodes(G, args):
    """
	Selects the most promising nodes from the graph	according to specified
	input arguments techniques.

	:param G:
	:param args:
	:return:
	"""
    # nodes filtering
    nodes = None
    if args["filter_best_spread_nodes"]:
        best_nodes = inverse_ncr(args["search_space_size_min"], args["k"])
        error = (inverse_ncr(args["search_space_size_max"], args["k"]) - best_nodes) / best_nodes
        filter_function = partial(monte_carlo_max_hop, G=G, random_generator=prng, p=args["p"], model=args["model"],
                                  max_hop=3, no_simulations=1)
        nodes = filter_best_spread_nodes(G, best_nodes, error, filter_function)

    nodes = filter_min_degree_nodes(G, args["min_degree"], nodes)

    return nodes


if __name__ == '__main__':
    
    filenames = ["scale_graphs/facebook_combined_scale_4.txt","graphs/facebook_combined.txt"]
    gt = ["comm_ground_truth/facebook_combined_4.csv","comm_ground_truth/facebook_combined.csv"]
    k_nodes = [4,1]
    models = ["IC","WC","LT"]
    i = 0
    for item in filenames:
        scale = k_nodes[i]
        file_gt = gt[i]
        i +=1
        filename = item
        for m in models:
            model = m

            G = read_graph(filename)

            print(nx.info(G))
            random_seed = 10
            prng = random.Random(random_seed)
            p = 0.1

            k = 100/ scale

           
            args = {}
            args["p"] = p
            args["model"] = model
            args["k"] = k
            args["filter_best_spread_nodes"] = False
            args["search_space_size_max"] = 100
            args["search_space_size_min"] = 10

            my_degree_function = G.degree
            mean = []
            for item in G:
                mean.append(my_degree_function[item])
            
            mean = int(np.mean(mean))  
            args["min_degree"] = mean + 1
            args["smart_initialization_percentage"] = 0.5
            args["population_size"] = 50
            nodes = filter_nodes(G, args)
            initial_population = create_initial_population(G, args, prng, nodes)

            communities =[]

            df = pd.read_csv(file_gt,sep=",")
            print(df)
            groups = df.groupby('comm')['node'].apply(list)
            print(groups)
            df = groups.reset_index(name='nodes')
            communities = df["nodes"].to_list()
            '''Propagation Simulation Parameters
            p: propability of activating node m when m is active and n-->m (only for IC Model)
            model: type of propagation model either IC (Indipendent Cascade) or WC(Weighted Cascade)
            no_simulations: number of simulations of the selected propagation model 
            '''

                    
            no_simulations = 50
            max_generations = 100
            #nodes' bound of seed sets
            #k=200
            #max_generations = 10 * k



            n_threads = 5
            
            #Print Graph's information and properties
            logging.info(nx.classes.function.info(G))
            
            
            #define file where to save the results obtained from the execution
            file = str(os.path.basename(filename))
            file = file.replace(".txt", "")
            file = '{0}-k{1}-p{2}-{3}.csv'.format(file, k, p , model)



            ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
            
            start = time.time()
            seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=50, offspring_size=50, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=MonteCarlo_simulation, population_file=file, nodes=nodes, communities=communities, initial_population=initial_population)
            
            exec_time = time.time() - start
            print(exec_time)
    
