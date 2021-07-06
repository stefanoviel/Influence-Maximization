import networkx as nx
import time
import random
from new_ea import *
import logging
import load

if __name__ == '__main__':
    #filename = 
    G = load.read_graph("graphs/ca-GrQc.txt")
    
    #nodes' bound
    k = 30 
        
    
    #influence propagation probability only for 'IC' model
    #p = 0.01
    p = 0.05

    ##Propagation Model
    model = 'IC'
    #model = 'WC'
    no_simulations = 100



    max_generations = 10 * k
    n_threads = 5
    random_seed = 10
    prng = random.Random()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO) # TODO switch between INFO and DEBUG for less or more in-depth logging
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 
 
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if random_seed == None: 
        random_seed = time()
    logging.info("Random number generator seeded with " + str(random_seed)+ " seed")
    prng.seed(random_seed)
    logging.info(prng)
	
    logging.info(nx.classes.function.info(G))
    #seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=50, offspring_size=50, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=spread.MonteCarlo_simulation)
    seed_sets = moea_influence_maximization(G, p, no_simulations, model, offspring_size=50, population_size=100, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=spread.MonteCarlo_simulation_max_hop, max_hop=10)
    logging.info("Seed sets {}".format(seed_sets))  