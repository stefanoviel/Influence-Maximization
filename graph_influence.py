import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from new_ea import *
from functions import progress
N = 200
def read_graph(filename, nodetype=int):    
    graph_class = nx.MultiGraph()
    G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)
    #G = nx.fast_gnp_random_graph(N,0.3)
    return G

if __name__ == '__main__':
    #filename = 
    G = read_graph("graphs/Twitch_EN.txt")
    k = 3
    p = 0.01
    model = 'WC'
    no_simulations = 100
    max_generations = 30
    n_threads = 2
    random_seed = 10
    prng = random.Random()
    import logging
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG) # TODO switch between INFO and DEBUG for less or more in-depth logging
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
    #seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=16, offspring_size=16, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=10, fitness_function=spread.MonteCarlo_simulation)
	
    logging.info(nx.classes.function.info(G))
    
    seed_sets = moea_influence_maximization(G, p, no_simulations, model,offspring_size=10, population_size=20,random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=10, fitness_function=spread.MonteCarlo_simulation_max_hop)
    #print(len(seed_sets))
    #print(str(seed_sets))
    #print(str(spread))
    logging.info("Seed sets {}".format(seed_sets))  
    print(type(seed_sets))
    for item in seed_sets:
        print(item)
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()
    #print(f'number of edges',G.number_of_edges())
    #for i in range(len(G)):
        #print(i)
        #print(G.edges(i,data=True))
        #print("\n\n")