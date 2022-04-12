"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import random
import logging
from time import time
# local libraries
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
from src.utils import to_csv
from src.ea.observer import hypervolume_observer
from src.ea.evaluator import nsga2_evaluator
from src.ea.generator import nsga2_generator
from src.ea.crossover import ea_one_point_crossover
from src.ea.terminators import generation_termination
from src.ea.mutators import ea_global_random_mutation
# inspyred libriaries
import inspyred
from inspyred.ec import *

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
    no_obj: number of objcetive function in the multi-objcetive optimization
    """
def moea_influence_maximization(G,args, fitness_function=None, fitness_function_kargs=dict(),random_gen=random.Random(),population_file=None, initial_population=None) :
    # initialize multi-objective evolutionary algorithm, NSGA-II
    nodes = list(G.nodes)
    
    if args["k"] == None : 
        max_seed_nodes = int(0.1 * len(nodes))
        logging.debug("Maximum size for the seed set has been set to %d" % max_seed_nodes)
    if population_file == None :
        population_file = "RandomGraph-N{nodes}-E{edges}-population.csv".format(nodes=len(G.nodes), edges=G.number_of_edges())
    if fitness_function == None :
        fitness_function = MonteCarlo_simulation
        fitness_function_kargs["random_generator"] = random_gen # pointer to pseudo-random number generator

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.variator = [ea_one_point_crossover,ea_global_random_mutation]
    ea.terminator = [generation_termination]
    ea.observer = [hypervolume_observer]
    
    #used the default NSGA-II replacer ec.replacers.nsga_replacement 
    #ea.replacer = inspyred.ec.replacers.generational_replacement
    #ea.replacer = inspyred.ec.replacers.plus_replacement
    
    bounder = inspyred.ec.DiscreteBounder(nodes)
    
    # start the evolutionary process
    ea.evolve(
        generator = nsga2_generator,
        evaluator = nsga2_evaluator,
        bounder= bounder,
        maximize = True,
        seeds = initial_population,
        pop_size = args["population_size"],
        num_selected = args["offspring_size"],
        generations_budget=args["max_generations"],
        #max_generations=int(max_generations*0.9), #no termination criteria used in this work
        tournament_size=args["tournament_size"],
        mutation_rate=args["mutation_rate"],
        crossover_rate=args["crossover_rate"],
        num_elites=args["num_elites"],
        # all arguments below will go inside the dictionary 'args'
        G = G,
        p = args["p"],
        model = args["model"],
        no_simulations = args["no_simulations"],
        nodes = nodes,
        n_threads = 1,
        min_seed_nodes = 1,
        max_seed_nodes = args["k"],
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
        fitness_function = fitness_function,
        fitness_function_kargs = fitness_function_kargs,
        mutation_operator=ea_global_random_mutation,
        graph = G,
        hypervolume = [], # keep track of HV trend throughout the generations
        no_obj = args["no_obj"], 
        time = [] # keep track of Time (Activation Attempts) trend throughout the generations
    )

    # extract seed sets from the final Pareto front/archive 
    if args["no_obj"] == 3:
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        to_csv(seed_sets, population_file)
    elif args["no_obj"]  == 2:
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1]] for individual in ea.archive] 
        args["no_obj"] (seed_sets, population_file)

    return seed_sets



