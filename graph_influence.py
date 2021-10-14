import os
import time
import random
import logging
import networkx as nx

# local libraries
import load
from new_ea import *

'''
File to launch a multi-objective evaluation of Influence Maximization 

- obj 1: number of reached nodes starting from a seed set (to maximize)
- obj 2: number of nodes in the seed set (to minimize)
- obj 3: time required to converge to the optima solution (to minimize)

This file in the main contains the parameters to be set for both the propagation model and the evolutionary algorithm used.

As a result the execution will print to the shell the seed sets of the best configuration found during the execution.

In addition a .csv file will be saved.
In this file, called name_graph.csv, you can find all the best solutions (by best we mean that they respect this MOEA).
It will be possible to find the nodes that form a seed set, their influence value, their size (number of nodes) and the time needed to converge.

REMINDER: the results in the csv are only those that form a Pareto Front using NSGA2.

'''

if __name__ == '__main__':
    

    filename = "prova"
    G = load.read_graph(filename)
    
    '''Propagation Simulation Parameters
    p: propability of activating node m when m is active and n-->m (only for IC Model)
    model: type of propagation model either IC (Indipendent Cascade) or WC(Weighted Cascade)
    no_simulations: number of simulations of the selected propagation model 
    '''
 
    p = 0.1
    #p = 0.05
    #p = 0.005

    model = 'LT'
    #model = 'WC'
            
    no_simulations = 100

    #nodes' bound of seed sets
    #k=200
    k = 50   
    max_generations = 10 * k



    n_threads = 1
    random_seed = 10
    prng = random.Random()
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG) # TODO switch between DEBUG and DEBUG for less or more in-depth logging
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 
 
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if random_seed == None: 
        random_seed = time()
    logging.info("Random number generator seeded with " + str(random_seed)+ " seed")
    prng.seed(random_seed)
    logging.info(prng)
	
    #Print Graph's information and properties
    logging.info(nx.classes.function.info(G))
    
    
    #define file where to save the results obtained from the execution
    file = str(os.path.basename(filename))
    file = file.replace(".txt", "")
    file = '{0}-k{1}-p{2}-{3}.csv'.format(file, k, p , model)
   

    ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
    seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=100, offspring_size=50, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=spread.MonteCarlo_simulation, population_file=file)
    
    ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION_MAX_HOP
    ##max_hop to define only in case FITNESS FUNCTION MONTECARLO_SIMULATION_MAX_HOP is chosen, otherwise default max_hop=2
    #max_hop = 10
    #seed_sets = moea_influence_maximization(G, p, no_simulations, model, offspring_size=50, population_size=100, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=spread.MonteCarlo_simulation_max_hop, max_hop=10,population_file=file)
    
    
    logging.info("Seed sets {}".format(seed_sets))  