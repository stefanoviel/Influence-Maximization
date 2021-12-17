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
    

    '''
    -1 !!!!!!!!!!!!
    terminators + here!
    
    '''
    #gt = ["comm_ground_truth/facebook_combined_4.csv","comm_ground_truth/facebook_combined_2.csv","comm_ground_truth/facebook_combined_1.33.csv","comm_ground_truth/facebook_combined.csv"]
    #filenames = ["scale_graphs/facebook_combined.txt_TRUE-4.txt","scale_graphs/facebook_combined.txt_TRUE-2.txt","scale_graphs/facebook_combined.txt_TRUE-1.33.txt","graphs/facebook_combined.txt"]

    #filenames = ["graphs/facebook_combined.txt"]
    #gt = ["comm_ground_truth/facebook_combined.csv"]
    filenames = ["scale_graphs/graph_SBM_small.txt_TRUE-4.txt","scale_graphs/graph_SBM_small.txt_TRUE-2.txt","scale_graphs/graph_SBM_small.txt_TRUE-1.33.txt","graphs/graph_SBM_small.txt"]

    gt = ["comm_ground_truth/graph_SBM_small_4.csv","comm_ground_truth/graph_SBM_small_2.csv","comm_ground_truth/graph_SBM_small_1.33.csv","comm_ground_truth/graph_SBM_small.csv"]
    #scale_k=[4,2,1.33,1]
    scale_k = [4,2,1.33,1]
    #scale_k = [1]
    models = ["IC"]


    #models = ['WC']
    i = 0
    for item in filenames:
        file_gt = gt[i]
        scale = scale_k[i]
        i +=1
        filename = item
        
        for m in models:
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
            t = "best_hv"
            if model == "IC":
                p = 0.05
            elif model == "LT":
                p = 0
            elif model == "IC_1":
                p = 0.1
                model = "IC"
            elif model == "WC":
                p = 0
            
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
            args["population_size"] = 50 

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
            nodes = filter_nodes(G, args)
            initial_population = create_initial_population(G, args, prng, nodes)


            no_obj = 2
            no_simulations = 50
            max_generations = 100
            #max_generations = 50
            #nodes' bound of seed sets
            #k=200
            #max_generations = 10 * k
            population_size = 50
            offspring_size = 50


            n_threads = 5
            


            # N_initial_population = int(50 * 1/scale) 
            # populations = []
            # hv_value = []
            # for ii in range(N_initial_population):
            #     initial_population = create_initial_population(G, args, prng, nodes)
            #     populations.append(initial_population)
            #     fitness = []               
            #     for iii in range(len(initial_population)):
            #         A_set = set(initial_population[iii])
            #         influence_mean, _, comm = MonteCarlo_simulation(G, A_set, p, no_simulations, model, communities, random_generator=None)
            #         s = []
            #         s.append(influence_mean)
            #         s.append(float(1.0 / len(A_set)))
            #         s.append(comm)
            #         fitness.append(s)
            #     for z in range(len(fitness)):
            #         for j in range(len(fitness[z])):
            #                 fitness[z][j] = -float(fitness[z][j])
                
            #     F =  np.array(fitness)

            #     t = (1/args["k"]) 
            #     tot = (G.number_of_nodes() -1) * (1 - t) * (len(communities) -1)
            
            #     from pymoo.indicators.hv import Hypervolume

            #     metric = Hypervolume(ref_point= np.array([-1,-t,-1]),
            #                         norm_ref_point=False,
            #                         zero_to_one=False)
            #     hv = metric.do(F)
            #     print(hv/tot)
            #     hv_value.append(hv/tot)

            # max_value = max(hv_value)

            # max_index = hv_value.index(max_value)

            #initial_population = populations[max_index]


            #Print Graph's information and properties
            logging.info(nx.classes.function.info(G))
            
            
            #define file where to save the results obtained from the execution
            file = str(os.path.basename(filename))
            file = file.replace(".txt", "")
            t = 'NEW_2_OBJ'
            file = '{0}-k{1}-p{2}-{3}-{4}'.format(file, k, p , model,t)
            ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
            start = time.time()
            seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=population_size, offspring_size=offspring_size, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=MonteCarlo_simulation, population_file=file, nodes=nodes, communities=communities, initial_population=initial_population ,no_obj=no_obj)
            
            exec_time = time.time() - start
            print(exec_time)
            exit(0)