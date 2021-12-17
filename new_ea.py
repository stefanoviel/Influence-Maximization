"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import random
import logging
import threading
from src.threadpool import ThreadPool
from time import time

# local libraries
from src.spread.monte_carlo import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
from src.ea.observer import ea_observer0, ea_observer1, ea_observer2

from src.ea.evaluator import nsga2_evaluator
from src.ea.crossover import ea_one_point_crossover
from src.ea.generator import nsga2_generator
from src.ea.generators import generator
from src.utils import to_csv, compute_time
from src.ea.mutators import ea_global_random_mutation
from src.ea.terminators import generation_termination,no_improvement_termination

# inspyred libriaries
import inspyred
from inspyred.ec import *
from inspyred.ec.emo import NSGA2

"""
Multi-objective evolutionary influence maximization. Parameters:
    G: networkx graph
    p: probability of influence spread 
    no_simulations: number of simulations
    model: type of influence propagation model
    population_size: population of the EA (default: value)
    offspring_size: offspring of the EA (default: value)
    max_generations: maximum generations (default: value)
    min_seed_nodes: minimum number of nodes in a seed set (default: 1)
    max_seed_nodes: maximum number of nodes in a seed set (default: 1% of the graph size)
    n_threads: number of threads to be used for concurrent evaluations (default: 1)
    random_gen: already initialized pseudo-random number generation
    initial_population: individuals (seed sets) to be added to the initial population (the rest will be randomly generated)
    population_file: name of the file that will be used to store the population at each generation (default: file named with date and time)
    max_hop: define the size of max_hop if fitness_function=MonteCarlo_max_hop has been selected
    """
def moea_influence_maximization(G, p, no_simulations, model, population_size=100, offspring_size=100, max_generations=100, min_seed_nodes=None, max_seed_nodes=None, n_threads=1, random_gen=random.Random(), population_file=None, fitness_function=None, fitness_function_kargs=dict(),max_hop=2, nodes=None, communities=None,initial_population=None) :
    # initialize multi-objective evolutionary algorithm, NSGA-II
    nodes = list(G.nodes)

    if min_seed_nodes == None :
        min_seed_nodes = 1
        logging.debug("Minimum size for the seed set has been set to %d" % min_seed_nodes)
    if max_seed_nodes == None : 
        max_seed_nodes = int(0.1 * len(nodes))
        logging.debug("Maximum size for the seed set has been set to %d" % max_seed_nodes)
    if population_file == None :
        population_file = "RandomGraph-N{nodes}-E{edges}-population.csv".format(nodes=len(G.nodes), edges=G.number_of_edges())
    if fitness_function == None :
        fitness_function = MonteCarlo_simulation
        fitness_function_kargs["random_generator"] = random_gen # pointer to pseudo-random number generator
    
    ea = inspyred.ec.emo.NSGA2(random_gen)
    
    
    #ea.observer = [ea_observer0, ea_observer1, ea_observer2] 
    ea.variator = [ea_one_point_crossover,ea_global_random_mutation]
    ea.terminator = [no_improvement_termination,generation_termination]
	
    #ea.replacer = inspyred.ec.replacers.generational_replacement
    #ea.replacer = inspyred.ec.replacers.plus_replacement
    bounder = inspyred.ec.DiscreteBounder(nodes)
    # start the evolutionary process
    ea.evolve(
        generator = nsga2_generator,
        evaluator = nsga2_evaluator,
        bounder= bounder,
        maximize = [True, False, True],
        seeds = initial_population,
        pop_size = population_size,
        num_selected = offspring_size,
        generations_budget=max_generations,
        max_generations=int(max_generations*0.15),
        tournament_size=5,
        mutation_rate=0.1,
        crossover_rate=0.1,
        num_elites=2,
        # all arguments below will go inside the dictionary 'args'
        G = G,
        p = p,
        model = model,
        no_simulations = no_simulations,
        nodes = nodes,
        n_threads = n_threads,
        min_seed_nodes = min_seed_nodes,
        max_seed_nodes = max_seed_nodes,
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
        fitness_function = fitness_function,
        fitness_function_kargs = fitness_function_kargs,
        mutation_operator=ea_global_random_mutation,
        communities = communities,
        graph = G,
        hypervolume = [],
        hv_influence_k = [],
        hv_influence_comm = [],
        hv_k_comm = []
    )

    # extract seed sets from the final Pareto front/archive

    #seed_sets = [[individual.candidate, individual.fitness[0], 1/ individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
  
    seed_sets = [[individual.candidate, individual.fitness[0],individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
  #seed_sets = [[individual.candidate, individual.fitness[0], 1/ individual.fitness[1]] for individual in ea.archive] 
    std, times = compute_time(seed_sets, population_file, G, model, p, no_simulations, communities, random_gen)
    to_csv(seed_sets, population_file, std, times)
    return seed_sets



