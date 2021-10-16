import os
import time
import random
import logging
import networkx as nx
from functools import partial

# local libraries
from src.load import read_graph
from src.spread import MonteCarlo_simulation, MonteCarlo_simulation_max_hop
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
    

    filename = "/Users/elia/Desktop/Influence-Maximization/graphs/facebook_combined.txt"
    G = read_graph(filename)

    print(nx.info(G))
    random_seed = 10
    prng = random.Random(random_seed)
    #p = 0.05
    p = 0.1

    #model = 'IC'
    model = 'WC'
    #model = 'LT'
    k = 5

    args = {}
    args["p"] = p
    args["model"] = model
    args["k"] = k
    args["filter_best_spread_nodes"] = False
    args["search_space_size_max"] = None
    args["search_space_size_min"] = None
    args["min_degree"] = 20
    args["smart_initialization_percentage"] = 0.5
    args["population_size"] = 10
    nodes = filter_nodes(G, args)

    #print(nodes)
    #print(G.number_of_nodes())
    #print(len(nodes))
    initial_population = create_initial_population(G, args, prng, nodes)

    communities = community_detection(G,10)

    '''Propagation Simulation Parameters
    p: propability of activating node m when m is active and n-->m (only for IC Model)
    model: type of propagation model either IC (Indipendent Cascade) or WC(Weighted Cascade)
    no_simulations: number of simulations of the selected propagation model 
    '''
 
            
    no_simulations = 5
    max_generations = 10
    #nodes' bound of seed sets
    #k=200
    #max_generations = 10 * k



    n_threads = 1
	
    #Print Graph's information and properties
    logging.info(nx.classes.function.info(G))
    
    
    #define file where to save the results obtained from the execution
    file = str(os.path.basename(filename))
    file = file.replace(".txt", "")
    file = '{0}-k{1}-p{2}-{3}.csv'.format(file, k, p , model)
   


    ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
    
    start = time.time()
    seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=10, offspring_size=10, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=MonteCarlo_simulation, population_file=file, nodes=nodes, communities=communities, initial_population=initial_population)
    
    exec_time = time.time() - start
    print(exec_time)
    
    
