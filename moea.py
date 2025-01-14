"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import random
import logging
from time import time
# local libraries
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
from src.utils import to_csv_influence_seedSize_time
from src.utils import to_csv_influence_seedSize_communities
from src.utils import to_csv_influence_seedSize_communities_time
from src.utils import to_csv_influence_communities_time
from src.utils import to_csv_influence_communities
from src.utils import to_csv_influence_time
from src.utils import to_csv2
from src.ea.observer import hypervolume_observer
from src.ea.observer import hypervolume_observer_all_combinations
from src.ea.evaluator import Nsga2
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
    """
def moea_influence_maximization(G,args, max_time, fitness_function=None, fitness_function_kargs=dict(),random_gen=random.Random(),population_file=None, initial_population=None) :
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
    
    comm = fitness_function_kargs["communities"]
    tot_comm = len(comm)

    fitness_function_kargs["max_time"] = max_time

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.variator = [ea_one_point_crossover,ea_global_random_mutation]
    ea.terminator = [generation_termination]
    ea.observer = [hypervolume_observer_all_combinations]
    
    #used the default NSGA-II replacer ec.replacers.nsga_replacement 
    #ea.replacer = inspyred.ec.replacers.generational_replacement
    #ea.replacer = inspyred.ec.replacers.plus_replacement
    
    bounder = inspyred.ec.DiscreteBounder(nodes)
    nsga2 = Nsga2()
    
    # start the evolutionary process
    ea.evolve(
        generator = nsga2_generator,
        evaluator = nsga2.nsga2_evaluator,
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
        elements_objective_function=args["elements_objective_function"], 
        num_elites=args["num_elites"],
        communities = comm,
        tot_communities = tot_comm,
        max_time = max_time, 
        # all arguments below will go inside the dictionary 'args'
        G = G,
        p = args["p"],
        model = args["model"],
        no_simulations = args["no_simulations"],
        nodes = nodes,
        n_threads = args["n_threads"],
        min_seed_nodes = 1,
        max_seed_nodes = args["k"],
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
        fitness_function = fitness_function,
        nsga2 = nsga2,  
        fitness_function_kargs = fitness_function_kargs,
        mutation_operator=ea_global_random_mutation,
        graph = G,
        hypervolume = [], # keep track of HV trend throughout the generations
        time = [] # keep track of Time (Activation Attempts) trend throughout the generations
    )

    # print([[i.fitness[0], i.fitness[1]] for i in ea.archive])

    # extract seed sets from the final Pareto front/archive 
    if args["elements_objective_function"] == "influence_seedSize_communities_time": 
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1], individual.fitness[2], individual.fitness[3]] for individual in ea.archive] 
        to_csv_influence_seedSize_communities_time(seed_sets, population_file)
    elif args["elements_objective_function"] == "influence_seedSize_time" :
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        to_csv_influence_seedSize_time(seed_sets, population_file)
    elif args["elements_objective_function"] == "influence_seedSize_communities": 
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        to_csv_influence_seedSize_communities(seed_sets, population_file)
    elif args["elements_objective_function"] == "influence_communities_time": 
        seed_sets = [[individual.candidate, individual.fitness[0] , individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        to_csv_influence_communities_time(seed_sets,population_file)
    elif args["elements_objective_function"] == "influence_time": 
        seed_sets = [[individual.candidate, individual.fitness[0] , individual.fitness[1]] for individual in ea.archive] 
        to_csv_influence_time(seed_sets, population_file)
    elif args["elements_objective_function"] == "influence_communities": 
        seed_sets = [[individual.candidate, individual.fitness[0] , individual.fitness[1]] for individual in ea.archive] 
        to_csv_influence_communities(seed_sets, population_file)
    elif args["elements_objective_function"] == "influence_seedSize": 
        seed_sets = [[individual.candidate, individual.fitness[0], ((args["k"]  / G.number_of_nodes()) * 100) - individual.fitness[1]] for individual in ea.archive] 
        to_csv2(seed_sets, population_file)

    return seed_sets



