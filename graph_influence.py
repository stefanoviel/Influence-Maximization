import os
import time
import random
import logging
import networkx as nx
from functools import partial
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
# local libraries
from src.load import read_graph
from src.spread.monte_carlo import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
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
        filter_function = partial(MonteCarlo_simulation_max_hop, G=G, random_generator=prng, p=args["p"], model=args["model"],
                                  max_hop=3, no_simulations=1)
        nodes = filter_best_spread_nodes(G, best_nodes, error, filter_function)

    nodes = filter_min_degree_nodes(G, args["min_degree"], nodes)

    return nodes


if __name__ == '__main__':
    

    no_runs = 10
    
    filenames = ["scale_graphs/facebook_combined_2.txt"]
    gt = ["comm_ground_truth/facebook_combined_2.csv"]
    
    scale_k = [2]

    #scale_k = [1]
    #models = ["IC"]


    models = ['IC']
    i = 0
    for item in filenames:
        file_gt = gt[i]
        scale = scale_k[i]
        i +=1
        filename = item
        file = str(os.path.basename(filename))
        file = file.replace(".txt", "")
        for m in models:
            if m == "IC":
                p = 0.05
            elif m == "LT":
                p = 0
            elif m == "WC":
                p = 0
            path = 'experiments/{0}-{1}'.format(file,m) 
            print(path)
            try:
                os.makedirs(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

            model = m
            print(model)
            G = read_graph(filename)

            print(nx.info(G))
            random_seed = 10
            prng = random.Random(random_seed)

            k = int(G.number_of_nodes() * 0.025)

            my_degree_function = G.degree
            mean = []
            for item in G:
                mean.append(my_degree_function[item])
            
            args = {}
            args["p"] = p
            args["model"] = model
            args["k"] = k
            args["filter_best_spread_nodes"] = True
            args["search_space_size_max"] = 1e11
            args["search_space_size_min"] = 1e9
            
            
            mean = int(np.mean(mean))  
            args["min_degree"] = mean + 1
            args["smart_initialization_percentage"] = 0.5
            args["population_size"] = 100

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
            nodes_original = filter_nodes(G, args)

            for r in range(no_runs):
                initial_population = create_initial_population(G, args, prng, nodes_original)

                no_obj = 3
                no_simulations = 100
                max_generations = 1000
                population_size = 100
                offspring_size = 100


                n_threads = 1
                


            

                #Print Graph's information and properties
                logging.info(nx.classes.function.info(G))
                
            
                file = 'run-{0}'.format(r+1)
                file = path+'/'+file
                print(file)
                #df.to_csv(path+'/'+file+'.csv')

                ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
                start = time.time()
                seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=population_size, offspring_size=offspring_size, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=MonteCarlo_simulation, population_file=file, communities=communities, initial_population=initial_population ,no_obj=no_obj)
                
                exec_time = time.time() - start   
                #print(exec_time)    
