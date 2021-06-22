"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import copy
import random
import logging
import collections
import inspyred

from time import time, strftime

from networkx.classes.function import edges

# local libraries
from functions import progress
import spread

# inspyred libriaries
from inspyred.ec import *
from inspyred.ec.emo import NSGA2
from inspyred.ec import EvolutionaryComputation
from inspyred.ec.replacers import nsga_replacement
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
def moea_influence_maximization(G, p, no_simulations, model, population_size=100, offspring_size=100, max_generations=100, min_seed_nodes=None, max_seed_nodes=None, n_threads=1, random_gen=random.Random(), initial_population=None, population_file=None, fitness_function=None, fitness_function_kargs=dict()) :

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

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.observer = ea_observer
    ea.variator = [nsga2_super_operator]
    ea.terminator = inspyred.ec.terminators.generation_termination
    
    # start the evolutionary process
    final_population = ea.evolve_1(
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

    # extract seed sets from the final Pareto front/archive
    seed_sets = [[individual.candidate, individual.fitness[0], 1/ individual.fitness[1], 1/individual.fitness[2] if individual.fitness[2]>0 else 0] for individual in ea.archive ] 
    #seed_sets = [ [individual.candidate, individual.fitness[0],1/individual.fitness[1] if individual.fitness[1]>0 else 0] for individual in ea.archive ] 
    print(len(seed_sets))
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
            fitness_function_args = [G, A_set, p, no_simulations, model]
            influence_mean, influence_std = fitness_function(*fitness_function_args, **fitness_function_kargs)
            gen = float(1/args["generations"] if args["generations"]>0 else 0)

            fitness[index] = inspyred.ec.emo.Pareto([influence_mean, (1.0 / float(len(A_set))), gen])
            #fitness[index] = inspyred.ec.emo.Pareto([influence_mean, gen]) 
            #print(fitness[index])
            #print(A)
            #print(type(fitness[index]))
        
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
            fitness_function_args = [G, A_set, p, no_simulations, model]           
            tasks.append((fitness_function, fitness_function_args, fitness_function_kargs, fitness, A_set, index, thread_lock, args["generations"]))

        thread_pool.map(nsga2_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    return fitness

#def nsga2_evaluator_threaded(G, p, A, no_simulations, model, fitness, index, thread_lock, thread_id) :
#
#    # TODO add logging?
#    A_set = set(A)
#    influence_mean, influence_std = spread.MonteCarlo_simulation_max_hop(G, A_set, p, no_simulations, model)
#
#    # lock data structure before writing in it
#    thread_lock.acquire()
#    fitness[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set))])  
#    thread_lock.release()
#
#    return 

def nsga2_evaluator_threaded(fitness_function, fitness_function_args, fitness_function_kargs, fitness_values, A_set, index, thread_lock, gen, thread_id) :

    #influence_mean, influence_std = spread.MonteCarlo_simulation_max_hop(G, A_set, p, no_simulations, model)
    influence_mean, influence_std = fitness_function(*fitness_function_args, **fitness_function_kargs)

    # lock data structure before writing in it
    thread_lock.acquire()
    gen = float(1/gen if gen>0 else 0)

    fitness_values[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set)), gen]) 
    thread_lock.release()

    return 

def ea_observer(archiver, num_generations, num_evaluations, args) :

    time_previous_generation = args['time_previous_generation']
    currentTime = time()
    timeElapsed = currentTime - time_previous_generation
    args['time_previous_generation'] = currentTime


    progress(args["generations"]+1, args["max_generations"], status='Generations{}'.format(args["generations"]+1))
    
    best = max(archiver)


    ## TO - PRINT IF NEEDED
    #logging.info('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))
    
    #for item in archiver:
        #logging.info('Candidate {cand} - Fitness {fit}'.format(cand=item.candidate, fit=item.fitness))

   
   
   
    # TODO write current state of the ALGORITHM to a file (e.g. random number generator, time elapsed, stuff like that)
    # write current state of the archiver to a file
    archiver_file = args["population_file"]
    # print("*********")
    # #print(best)
    # print(archiver[0])
    # print(archiver[0].fitness[0])
    # print(archiver[0].fitness[1])
    # print(archiver[0].fitness[2])

    # print(archiver[0].candidate)

    # print("*******")
    #find the longest individual
    max_length = len(max(archiver, key=lambda x : len(x.candidate)).candidate)

    with open(archiver_file, "w") as fp :
        # header, of length equal to the maximum individual length in the archiver
        fp.write("n_nodes,influence,generations")
        #fp.write("influence,generations")

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
    
    # return the two children
    if len(child1) > 0 and len(child1) <= max_seed_nodes : children.append( child1 )
    if len(child2) > 0 and len(child2) <= max_seed_nodes : children.append( child2 )

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

def evolve_2(self,pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        """Perform the evolution.
        
        This function creates a population and then runs it through a series
        of evolutionary epochs until the terminator is satisfied. The general
        outline of an epoch is selection, variation, evaluation, replacement,
        migration, archival, and observation. The function returns a list of
        elements of type ``Individual`` representing the individuals contained
        in the final population.
        
        Arguments:
        
        - *generator* -- the function to be used to generate candidate solutions 
        - *evaluator* -- the function to be used to evaluate candidate solutions
        - *pop_size* -- the number of Individuals in the population (default 100)
        - *seeds* -- an iterable collection of candidate solutions to include
          in the initial population (default None)
        - *maximize* -- Boolean value stating use of maximization (default True)
        - *bounder* -- a function used to bound candidate solutions (default None)
        - *args* -- a dictionary of keyword arguments

        The *bounder* parameter, if left as ``None``, will be initialized to a
        default ``Bounder`` object that performs no bounding on candidates.
        Note that the *_kwargs* class variable will be initialized to the *args* 
        parameter here. It will also be modified to include the following 'built-in' 
        keyword argument:
        
        - *_ec* -- the evolutionary computation (this object)
        
        """
        self._kwargs = args
        self._kwargs['_ec'] = self
        self._kwargs["generations"] = 0
        if seeds is None:
            seeds = []
        if bounder is None:
            bounder = Bounder()
        
        self.termination_cause = None
        self.bounder = bounder
        self.maximize = maximize
        self.population = []
        self.archive = []
        self.replacer = nsga_replacement
        generator= nsga2_generator
        evaluator = nsga2_evaluator
        # Create the initial population.

        #print(self.variator)
        if not isinstance(seeds, collections.Sequence):
            seeds = [seeds]
        initial_cs = copy.copy(seeds)
        num_generated = max(pop_size - len(seeds), 0)
        i = 0
        self.logger.debug('generating initial population')
        while i < num_generated:
            cs = generator(random=self._random, args=self._kwargs)
            initial_cs.append(cs)
            i += 1
        self.logger.debug('evaluating initial population')
        initial_fit = evaluator(candidates=initial_cs, args=self._kwargs)
        
        for cs, fit in zip(initial_cs, initial_fit):
            if fit is not None:
                ind = Individual(cs, maximize=maximize)
                ind.fitness = fit
                self.population.append(ind)
            else:
                self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
        self.logger.debug('population size is now {0}'.format(len(self.population)))
        
        self.num_evaluations = len(initial_fit)
        self.num_generations = 0
        self._kwargs["generations"] = self.num_generations
        
        self.logger.debug('archiving initial population')
        self.archive = self.archiver(random=self._random, population=list(self.population), archive=list(self.archive), args=self._kwargs)
        self.logger.debug('archive size is now {0}'.format(len(self.archive)))
        self.logger.debug('population size is now {0}'.format(len(self.population)))
                
        if isinstance(self.observer, collections.Iterable):
            for obs in self.observer:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                obs(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        else:
            self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
            self.observer(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        
        while not self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            # Select individuals.
            self.logger.debug('selection using {0} at generation {1} and evaluation {2}'.format(self.selector.__name__, self.num_generations, self.num_evaluations))
            parents = self.selector(random=self._random, population=list(self.population), args=self._kwargs)
            self.logger.debug('selected {0} candidates'.format(len(parents)))
            parent_cs = [copy.deepcopy(i.candidate) for i in parents]
            offspring_cs = parent_cs
            
            if isinstance(self.variator, collections.Iterable):
                for op in self.variator:
                    self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(op.__name__, self.num_generations, self.num_evaluations))
                    offspring_cs = op(random=self._random, candidates=offspring_cs, args=self._kwargs)
            else:
                self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(self.variator.__name__, self.num_generations, self.num_evaluations))
                offspring_cs = self.variator(random=self._random, candidates=offspring_cs, args=self._kwargs)
            self.logger.debug('created {0} offspring'.format(len(offspring_cs)))
            
            # Evaluate offspring.
            self.logger.debug('evaluation using {0} at generation {1} and evaluation {2}'.format(evaluator.__name__, self.num_generations, self.num_evaluations))
            self._kwargs["generations"] = self.num_generations

            offspring_fit = evaluator(candidates=offspring_cs, args=self._kwargs)
            offspring = []
            for cs, fit in zip(offspring_cs, offspring_fit):
                if fit is not None:
                    off = Individual(cs, maximize=maximize)
                    off.fitness = fit
                    offspring.append(off)
                else:
                    self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
            self.num_evaluations += len(offspring_fit)        
            #print(self.replacer)
            # Replace individuals.
            self.logger.debug('replacement using {0} at generation {1} and evaluation {2}'.format(self.replacer.__name__, self.num_generations, self.num_evaluations))
            self.population = self.replacer(random=self._random, population=self.population, parents=parents, offspring=offspring, args=self._kwargs)
            self.logger.debug('population size is now {0}'.format(len(self.population)))
            # Migrate individuals.
            self.logger.debug('migration using {0} at generation {1} and evaluation {2}'.format(self.migrator.__name__, self.num_generations, self.num_evaluations))
            self.population = self.migrator(random=self._random, population=self.population, args=self._kwargs)
            self.logger.debug('population size is now {0}'.format(len(self.population)))
            
            # Archive individuals.
            self.logger.debug('archival using {0} at generation {1} and evaluation {2}'.format(self.archiver.__name__, self.num_generations, self.num_evaluations))
            self.archive = self.archiver(random=self._random, archive=self.archive, population=list(self.population), args=self._kwargs)
            self.logger.debug('archive size is now {0}'.format(len(self.archive)))
            self.logger.debug('population size is now {0}'.format(len(self.population)))

            print('archive size is now {0}'.format(len(self.archive)))
            print('population size is now {0}'.format(len(self.population)))           
            self.num_generations += 1
            if isinstance(self.observer, collections.Iterable):
                for obs in self.observer:
                    self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                    obs(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
            else:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
                self.observer(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        return self.population

EvolutionaryComputation.evolve_2 = evolve_2

def evolve_1(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tournament_size', 2)
        return EvolutionaryComputation.evolve_2(self, pop_size, seeds, maximize, bounder, **args)

NSGA2.evolve_1 = evolve_1

# if __name__ == "__main__" :

#     # initialize logging
#     import logging
#     logger = logging.getLogger('')
#     logger.setLevel(logging.DEBUG) # TODO switch between INFO and DEBUG for less or more in-depth logging
#     formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 
 
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     import load
#     k = 30
#     G = load.read_graph("graphs/Email_URV.txt")
#     p = 0.01
#     model = 'WC'
#     no_simulations = 100
#     max_generations = 10
#     n_threads = 1
#     random_seed = 42

#     prng = random.Random()
#     if random_seed == None: 
#         random_seed = time()
    
#     logging.debug("Random number nsga2_generator seeded with %s" % str(random_seed))
#     prng.seed(random_seed)

#     # try to pass max_seed_nodes=k to moea:
#     seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=16, offspring_size=16, random_gen=prng, max_generations=max_generations, n_threads=n_threads, max_seed_nodes=k, fitness_function=spread.MonteCarlo_simulation)
#     #seed_sets, spread = ea_influence_maximization(k, G, p, no_simulations, model, population_size=16, offspring_size=16, random_gen=prng, max_generations=max_generations, n_threads=n_threads)

#     logging.debug("Seed sets:")
#     logging.debug(str(seed_sets))
