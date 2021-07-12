"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import copy
import random
import logging
import collections
import inspyred

from time import time


# local libraries
from functions import progress
import spread 

# inspyred libriaries
from inspyred.ec import *
import override
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
    """
def moea_influence_maximization(G, p, no_simulations, model, population_size=100, offspring_size=100, max_generations=100, min_seed_nodes=None, max_seed_nodes=None, n_threads=1, random_gen=random.Random(), initial_population=None, population_file=None, fitness_function=None, fitness_function_kargs=dict(), max_hop=2) :

    # initialize multi-objective evolutionary algorithm, NSGA-II
    logging.debug("Setting up NSGA-II...")
    progress(0, max_generations, status='Inizializing')

    # check if some of the parameters are set; otherwise, use default values
    nodes = list(G.nodes)
    if min_seed_nodes == None :
        min_seed_nodes = 1
        logging.debug("Minimum size for the seed set has been set to %d" % min_seed_nodes)
    if max_seed_nodes == None : 
        max_seed_nodes = int( 0.1 * len(nodes))
        logging.debug("Maximum size for the seed set has been set to %d" % max_seed_nodes)
    if population_file == None :
        #ct = time()
        population_file = "RandomGraph-N{nodes}-E{edges}-population.csv".format(nodes=len(G.nodes), edges=G.number_of_edges())
    if fitness_function == None :
        fitness_function = spread.MonteCarlo_simulation_max_hop
        fitness_function_kargs["random_generator"] = random_gen # pointer to pseudo-random number generator
        logging.info("Fitness function not specified, defaulting to \"%s\"" % fitness_function.__name__)
    else :
        logging.info("Fitness function specified, \"%s\"" % fitness_function.__name__)
    if max_hop == None:
        max_hop = 2
    EvolutionaryComputation.new_evolve_ea = override.new_evolve_ea
    NSGA2.new_evolve_nsga2 = override.new_evolve_nsga2

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.observer = ea_observer
    ea.variator = [nsga2_super_operator]
    ea.terminator = inspyred.ec.terminators.generation_termination

    # start the evolutionary process
    if fitness_function != spread.MonteCarlo_simulation_max_hop:
        ea.new_evolve_nsga2(
            generator = nsga2_generator,
            evaluator = nsga2_evaluator,
            maximize = True,
            seeds = initial_population,
            pop_size = population_size,
            num_selected = offspring_size,
            max_generations = max_generations,
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
        )
    else:
        ea.new_evolve_nsga2(
            generator = nsga2_generator,
            evaluator = nsga2_evaluator,
            maximize = True,
            seeds = initial_population,
            pop_size = population_size,
            num_selected = offspring_size,
            max_generations = max_generations,
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
            max_hop = max_hop,
           )

    # extract seed sets from the final Pareto front/archive
    seed_sets = [[individual.candidate, individual.fitness[0], 1/ individual.fitness[1], 1/individual.fitness[2] if individual.fitness[2]>0 else 0] for individual in ea.archive ] 
    return seed_sets

def nsga2_evaluator(candidates, args):
    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]
    random_generator = args["_ec"]._random 

    fitness_function = args["fitness_function"]
    fitness_function_kargs = args["fitness_function_kargs"]
    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 
    #print(candidates)
    if n_threads == 1 :
        for index, A in enumerate(candidates) :

            # TODO sort phenotype, use cache...? or manage sorting directly during individual creation? 
            # TODO see lines 108-142 in src_OLD/multiObjective-inspyred/sn-inflmax-inspyred.py 
            # TODO maybe if we make sure that candidates are already sets before getting here, we could save some computational time
            # TODO consider std inside the fitness in some way?
            A_set = set(A)

            #influence_mean, influence_std = spread.MonteCarlo_simulation_max_hop(G, A_set, p, no_simulations, model, random_generator=random_generator)

            # NOTE now passing a generic function works, but the whole thing has to be implemented for the multi-threaded version
            if fitness_function != spread.MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            influence_mean, influence_std, time = fitness_function(*fitness_function_args, **fitness_function_kargs)
            #gen = float(1/args["generations"] if args["generations"]>0 else 0)
            time = float(1/time if time>0 else 0)
            fitness[index] = inspyred.ec.emo.Pareto([influence_mean, (1.0 / float(len(A_set))), time])
        
    else :
        
        # create a threadpool, using the local module
        import threadpool
        thread_pool = threadpool.ThreadPool(n_threads)

        # create thread lock, to be used for concurrency
        import threading
        thread_lock = threading.Lock()

        # create list of tasks for the thread pool, using the threaded evaluation function
        #tasks = [ (G, p, A, no_simulations, model, fitness, index, thread_lock) for index, A in enumerate(candidates) ]
        tasks = []
        for index, A in enumerate(candidates) :
            A_set = set(A)
            if fitness_function != spread.MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            tasks.append((fitness_function, fitness_function_args, fitness_function_kargs, fitness, A_set, index, thread_lock, args["generations"]))

        thread_pool.map(nsga2_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    return fitness


def nsga2_evaluator_threaded(fitness_function, fitness_function_args, fitness_function_kargs, fitness_values, A_set, index, thread_lock, gen, thread_id) :

    #influence_mean, influence_std = spread.MonteCarlo_simulation_max_hop(G, A_set, p, no_simulations, model)
    influence_mean, influence_std, time = fitness_function(*fitness_function_args, **fitness_function_kargs)
    time = float(1/time if time>0 else 0)
    # lock data structure before writing in it
    thread_lock.acquire()
    gen = float(1/gen if gen>0 else 0)

    #fitness_values[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set)), gen]) 
    fitness_values[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set)), time]) 
    print(fitness_values[index])
    thread_lock.release()

    return 

def ea_observer(archiver, num_generations, num_evaluations, args) :

    currentTime = time()
    args['time_previous_generation'] = currentTime


    progress(args["generations"]+1, args["max_generations"], status='Generations{}'.format(args["generations"]+1))
    
    ## TO - PRINT IF NEEDED
    #logging.info('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))
    
    #for item in archiver:
        #logging.info('Candidate {cand} - Fitness {fit}'.format(cand=item.candidate, fit=item.fitness))

   
   
   
    # TODO write current state of the ALGORITHM to a file (e.g. random number generator, time elapsed, stuff like that)
    # write current state of the archiver to a file
    archiver_file = args["population_file"]

    #find the longest individual
    max_length = len(max(archiver, key=lambda x : len(x.candidate)).candidate)

    with open(archiver_file, "w") as fp :
        # header, of length equal to the maximum individual length in the archiver
        #fp.write("n_nodes,influence,generations")

        fp.write("n_nodes,influence,n_simulation")

        for i in range(0, max_length) : fp.write(",n%d" % i)
        fp.write("\n")

        # and now, we write stuff, individual by individual
        for individual in  archiver :

            # check if fitness is an iterable collection (e.g. a list) or just a single value
            if hasattr(individual.fitness, "__iter__") :
                gen =  float(1.0 / individual.fitness[2] if individual.fitness[2] > 0 else 0)
                #gen =  float(1.0 / individual.fitness[1] if individual.fitness[1] > 0 else 0)

                fp.write("%d,%.4f,%d" % (1.0 / individual.fitness[1], individual.fitness[0], gen))
                #fp.write("%.4f,%d" % (individual.fitness[0], gen))

            else :
                fp.write("%d,%.4f" % (len(set(individual.candidate)), individual.fitness))

            for node in individual.candidate :
                fp.write(",%d" % node)

            for i in range(len(individual.candidate), max_length - len(individual.candidate)) :
                fp.write(",")

            fp.write("\n")

# TODO is there a way to have a multi-threaded generation of individuals?
@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2_super_operator(random, candidate1, candidate2, args) :
  
    children = []

    # uniform choice of operator
    randomChoice = random.randint(0,3)
    #randomChoice = 0

    if randomChoice == 0 :
        children = nsga2_crossover(random, list(candidate1), list(candidate2), args)
    elif randomChoice == 1 :
        children.append( ea_alteration_mutation(random, list(candidate1), args) )
    elif randomChoice == 2 :
        children.append( nsga2_insertion_mutation(random, list(candidate1), args) )
    elif randomChoice == 3 :
        children.append( nsga2_removal_mutation(random, list(candidate1), args) )

    # purge the children from "None" and empty arrays
    children = [c for c in children if c is not None and len(c) > 0]
    
    # this should probably be commented or sent to logging
    for c in children : logging.debug("randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)) )
    
    return children

#@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover
def nsga2_crossover(random, candidate1, candidate2, args): 

    children = []   
    max_seed_nodes = args["max_seed_nodes"]

    parent1 = list(set(candidate1))
    parent2 = list(set(candidate2))

    #print('Parent 1 {0} \nParent 2 {1} \n'.format(candidate1, candidate2))


    # choose random cut point
    cutPoint1 = random.randint(0, len(parent1)-1)
    cutPoint2 = random.randint(0, len(parent2)-1)
    
    # children start as empty lists
    child1 = []
    child2 = []
    
    # swap stuff
    for i in range(0, cutPoint1) : child1.append( parent1[i] )
    for i in range(0, cutPoint2) : child2.append( parent2[i] )
    
    for i in range(cutPoint1, len(parent2)) : child1.append( parent2[i] )
    for i in range(cutPoint2, len(parent1)) : child2.append( parent1[i] )
    
    # reduce children to minimal form
    child1 = list(set(child1))
    child2 = list(set(child2))
    #('Child 1 {0} \nChild 2 {1} \n'.format(child1, child2))

    # return the two children
    if len(child1) > 0 and len(child1) <= max_seed_nodes : children.append( child1 )
    if len(child2) > 0 and len(child2) <= max_seed_nodes : children.append( child2 )

    #print('Final Children {0}\n'.format(children))

    return children

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def ea_alteration_mutation(random, candidate, args) :
    
    #print("nsga2alterationMutation received this candidate:", candidate)
    nodes = args["nodes"]

    mutatedIndividual = list(set(candidate))

    # choose random place
    gene = random.randint(0, len(mutatedIndividual)-1)
    mutatedIndividual[gene] = nodes[ random.randint(0, len(nodes)-1) ]

    return mutatedIndividual

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2_insertion_mutation(random, candidate, args) :
    
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    mutatedIndividual = list(set(candidate))

    if len(mutatedIndividual) < max_seed_nodes :
        mutatedIndividual.append( nodes[ random.randint(0, len(nodes)-1) ] )
        return mutatedIndividual
    else :
        return None

# TODO take into account minimal seed set size
#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2_removal_mutation(random, candidate, args) :
    
    mutatedIndividual = list(set(candidate))

    if len(candidate) > 1 :
        gene = random.randint(0, len(mutatedIndividual)-1)
        mutatedIndividual.pop(gene)
        return mutatedIndividual
    else :
        return None

@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def nsga2_generator(random, args) :

    min_seed_nodes = args["min_seed_nodes"]
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    logging.debug("Min seed set size: %d; Max seed set size: %d" % (min_seed_nodes, max_seed_nodes))

    # extract random number in 1,max_seed_nodes
    individual_size = random.randint(min_seed_nodes, max_seed_nodes)
    individual = [0] * individual_size
    logging.debug( "Creating individual of size %d, with genes ranging from %d to %d" % (individual_size, nodes[0], nodes[-1]) )
    for i in range(0, individual_size) : individual[i] = nodes[ random.randint(0, len(nodes)-1) ]
    logging.debug(individual)

    return individual

