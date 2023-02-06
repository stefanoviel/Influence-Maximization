import os
import logging
import numpy as np
import inspyred.ec
import pandas as pd
from pymoo.indicators.hv import Hypervolume
#local libraries
from src.ea.generators import generator_new_nodes
from src.utils import diversity, individuals_diversity

def adjust_population_size(num_generations, population, args):
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness

	improvement = (current_best - prev_best) / prev_best

	if "improvement" in args.keys():
		args["improvement"].pop(0)
		args["improvement"].append(improvement)
	else:
		args["improvement"] = [0] * 3
	if len(population) < 100:
		if sum(args["improvement"]) == 0 and num_generations > 2:
			# new_individuals = min(int(1/div), 10)
			new_individuals = 1
			for _ in range(new_individuals):
				# candidate = args["_ec"].generator(args["prng"], args)
				candidate = generator_new_nodes(args["prng"], args)
				args["_ec"].population.append(inspyred.ec.Individual(candidate=candidate))
				args["_ec"].population[-1].fitness = args["fitness_function"](A=candidate)[0]
			args["num_selected"] = len(args["_ec"].population)

		elif len(population) > args["min_pop_size"] and improvement > 0:
			min_fit_ind = args["_ec"].population[0]
			for ind in args["_ec"].population:
				if ind.fitness < min_fit_ind.fitness:
					min_fit_ind = ind
			args["_ec"].population.remove(min_fit_ind)
			args["num_selected"] = len(args["_ec"].population)


def ea_observer0(population, num_generations, num_evaluations, args):
	"""
	adjusting some dynamic parameters of the dynamic algorithms
	"""
	# check for repetitions
	for ind in population:
		if len(set(ind.candidate)) != len(ind.candidate):
			raise NameError("Nodes repetition inside an individual")
	# exploration weight exponential decay
	if args["mab"] is not None:
		args["mab"].exploration_weight = 1 / (num_generations + 1) ** (3)
		logging.debug("Mab selections: {}".format(args["mab"].n_selections))
	logging.debug("Population size: {}".format(len(population)))

	if args["dynamic_population"]:
		adjust_population_size(num_generations, population, args)


def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, # printing to stdout some generational info
	"""
	# to access to evolutionary computation stuff
	div = diversity(population)
	logging.debug("generation {}: diversity {}".format(num_generations, div))
	ind_div = individuals_diversity(population)
	logging.debug("generation {}: individuals diversity {}".format(num_generations, ind_div))

	return


def ea_observer2(population, num_generations, num_evaluations, args):
	"""
	# printing generational log to out files
	"""

	# write current state of the population to a file
	sf = args["statistics_file"]

	# compute some generation statistics
	generation_stats = {}
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness
	if prev_best > 0:
		generation_stats["improvement"] = (current_best - prev_best) / prev_best
	else:
		generation_stats["improvement"] = 0
	args["prev_population_best"] = current_best

	sf.seek(sf.tell()-1, os.SEEK_SET)

	sf.write(",{},".format(diversity(population)))
	sf.write("{},".format(generation_stats["improvement"]))
	if args["mab"] is not None:
		sf.write("{}\n".format(args["mab"].n_selections))
	else:
		sf.write("\n")

	return



def time_observer(population, num_generations, num_evaluations, args):
    
	"""
	Save Time (Activation Attempts) at the end of the evolutionary process.
	"""

	df = pd.DataFrame(args["time"])
	df.to_csv(args["population_file"] + '-time.csv', index=False, header=None)
	return 


def hypervolume_observer(population, num_generations, num_evaluations, args):
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    arch = [list(x.fitness) for x in args["_ec"].archive] 
    for i in range(len(arch)):
        for j in range(len(arch[i])):
                arch[i][j] = -float(arch[i][j])
    F =  np.array(arch)

    if args["no_obj"] == 3:  # TODO fix 
        tot = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) * (len(args["communities"])-1)
        metric = Hypervolume(ref_point= np.array([0,0,-1]),
                            norm_ref_point=False,
                            zero_to_one=False)
        hv = metric.do(F)
        current_best = hv/tot
        args["hypervolume"].append(current_best)
        logging.info('Hypervolume at generation {0} : {1} '.format(num_generations,current_best))
    elif args["no_obj"] == 2:
        tot = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) 
        metric = Hypervolume(ref_point= np.array([0,0]),
                            norm_ref_point=False,
                            zero_to_one=False)
        hv = metric.do(F)
        current_best = hv/tot
        args["hypervolume"].append(current_best)
        logging.info('Hypervolume at generation {0} : {1} '.format(num_generations,current_best))


def get_elements_PF(PF, all_individuals): 
    pass



def hypervolume_observer_all_combinations(population, num_generations, num_evaluations, args):
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    arch = [list(x.fitness) for x in args["_ec"].archive] 
    candidates = [set(x.candidate) for x in args["_ec"].archive] 

    df = args["nsga2"].df
    df_com_time = pd.DataFrame(columns=['set', 'communities', 'time'])

    for c in candidates: 
        elems = df.loc[df['set'] == c]
        if len(elems) > 1: 
            # print(elems)
            elems = elems.iloc[0]
            # print(elems)
            df_com_time.loc[len(df_com_time), df_com_time.columns] = elems

        else: 
            df_com_time.loc[len(df_com_time), df_com_time.columns] = elems.iloc[0]

    df_com_time['time'] = df_com_time['time'].apply(lambda a: -a)
    df_com_time['communities'] = df_com_time['communities'].apply(lambda a: -a)
    # # print(df_com_time)
    
    for i in range(len(arch)):
        for j in range(len(arch[i])):
                arch[i][j] = -float(arch[i][j])
    F =  np.array(arch)

    # print(len(F), len(df_com_time))

    
    tot_influence_seed = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) 
    metric_influence_seed = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)

    tot_influence_time = 100 * 1
    metric_influence_time = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)

    tot_seed_time = ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100)  * 1
    metric_seed_time = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)

    tot_influence_seedSize_time = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) * 1   	# makes it explicit we are taking into account the max value for time (1)
    metric_influence_seedSize_time = Hypervolume(ref_point= np.array([0,0,0]),
                            norm_ref_point=False,
                            zero_to_one=False)
    
    tot_influence_communities = 100 * (len(args["communities"])-1) 
    metric_influence_communities = Hypervolume(ref_point= np.array([0,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)

    tot_seed_communities = ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100)  * (len(args["communities"])-1) 
    metric_seed_communities = Hypervolume(ref_point= np.array([0,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)

    tot_influence_seedSize_communities = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) * (len(args["communities"])-1) 
    metric_influence_seedSize_communities = Hypervolume(ref_point= np.array([0,0, -1]),
                            norm_ref_point=False,
                            zero_to_one=False)

    tot_influence_seedSize_communities_time = 100 * ((args["max_seed_nodes"] / args["graph"].number_of_nodes()) * 100) * (len(args["communities"])-1) * 1	# makes it explicit we are taking into account the max value for time (1)
    metric_influence_seedSize_communities_time = Hypervolume(ref_point= np.array([0,0, -1, 0]),
                        norm_ref_point=False,
                        zero_to_one=False)

    if args["elements_objective_function"] == "influence_seedSize": 

        hv_influence_seed = metric_influence_seed(F)
        hv_influence_seed = hv_influence_seed/tot_influence_seed

        hv_influence_seedSize_time = metric_influence_seedSize_time(np.concatenate([F, np.transpose([df_com_time['time']])], axis=1))
        hv_influence_seedSize_time = hv_influence_seedSize_time/tot_influence_seedSize_time
        # # print("\thv_influence_seedSize_time:", hv_influence_seedSize_time)  

        hv_influence_seedSize_communities = metric_influence_seedSize_communities(np.concatenate([F, np.transpose([df_com_time['communities']])], axis=1))
        hv_influence_seedSize_communities = hv_influence_seedSize_communities/tot_influence_seedSize_communities
        # # print("\thv_influence_seedSize_communities:", hv_influence_seedSize_communities)

        args["hypervolume"].append([hv_influence_seed, hv_influence_seedSize_time, hv_influence_seedSize_communities])
        # # print("\thv_influence_seed:", hv_influence_seed)

    elif args["elements_objective_function"] == "influence_seedSize_time": 

        hv_influence_seedSize_time = metric_influence_seedSize_time(F)
        hv_influence_seedSize_time = hv_influence_seedSize_time/tot_influence_seedSize_time
        # # print("\thv_influence_seedSize_time:", hv_influence_seedSize_time)

        hv_influence_seedSize_communities = metric_influence_seedSize_communities(np.concatenate([F[:,:-1], np.transpose([df_com_time['communities']])], axis=1))
        hv_influence_seedSize_communities = hv_influence_seedSize_communities/tot_influence_seedSize_communities
        # print("\thv_influence_seedSize_communities:", hv_influence_seedSize_communities)

        hv_influence_time = metric_influence_time( np.concatenate([np.transpose([F[:, 0]]) , np.transpose( [F[:, 2]]) ], axis=1))
        hv_influence_time = hv_influence_time/tot_influence_time
        # print("\thv_influence_time:", hv_influence_time)

        hv_seed_time = metric_seed_time(F[:, 1:])
        hv_seed_time = hv_seed_time/tot_seed_time
        # print("\thv_seed_time:", hv_seed_time)

        hv_influence_seed = metric_influence_seed(F[:, :2])
        hv_influence_seed = hv_influence_seed/tot_influence_seed
        # print("\thv_influence_seed:", hv_influence_seed)

        args["hypervolume"].append([hv_influence_seed, hv_influence_seedSize_time, hv_influence_seedSize_communities, hv_influence_time, hv_seed_time])

    elif args["elements_objective_function"] == "influence_seedSize_communities": 

        hv_influence_seedSize_communities = metric_influence_seedSize_communities(F)
        hv_influence_seedSize_communities = hv_influence_seedSize_communities/tot_influence_seedSize_communities
        # print("\thv_influence_seedSize_communities:", hv_influence_seedSize_communities)

        hv_influence_seedSize_time = metric_influence_seedSize_time(np.concatenate([F[:,:-1], np.transpose([df_com_time['time']])], axis=1))
        hv_influence_seedSize_time = hv_influence_seedSize_time/tot_influence_seedSize_time
        # print("\thv_influence_seedSize_time:", hv_influence_seedSize_time) 

        hv_influence_communities = metric_influence_communities(np.concatenate([np.transpose([F[:, 0]]) , np.transpose( [F[:, 2]]) ], axis=1))
        hv_influence_communities = hv_influence_communities/tot_influence_communities
        # print("\thv_influence_communities:", hv_influence_communities)

        hv_seed_communities = metric_seed_communities(F[:, 1:])
        hv_seed_communities = hv_seed_communities/tot_seed_communities
        # print("\thv_seed_communities:", hv_seed_communities)

        hv_influence_seed = metric_influence_seed(F[:, :2])
        hv_influence_seed = hv_influence_seed/tot_influence_seed
        # print("\thv_influence_seed:", hv_influence_seed)

        args["hypervolume"].append([hv_influence_seed, hv_influence_seedSize_time, hv_influence_seedSize_communities, 
         hv_influence_communities, hv_seed_communities])

    elif args["elements_objective_function"] == "influence_seedSize_communities_time": 
        
        hv_influence_seedSize_communities_time = metric_influence_seedSize_communities_time(F)
        hv_influence_seedSize_communities_time = hv_influence_seedSize_communities_time/tot_influence_seedSize_communities_time
        # print("\thv_influence_seedSize_communities_time:", hv_influence_seedSize_communities_time)


        hv_influence_communities = metric_influence_communities(  np.concatenate([np.transpose([F[:, 0]]) , np.transpose( [F[:, 2]]) ], axis=1))
        hv_influence_communities = hv_influence_communities/tot_influence_communities
        # print("\thv_influence_communities:", hv_influence_communities)

        hv_seed_communities = metric_seed_communities(F[:, 1:3])
        hv_seed_communities = hv_seed_communities/tot_seed_communities
        # print("\thv_seed_communities:", hv_seed_communities)

        hv_influence_time = metric_influence_time( np.concatenate([np.transpose([F[:, 0]]) , np.transpose( [F[:, 3]]) ], axis=1))
        hv_influence_time = hv_influence_time/tot_influence_time
        # print("\thv_influence_time:", hv_influence_time)

        hv_seed_time = metric_seed_time( np.concatenate([np.transpose([F[:, 1]]) , np.transpose( [F[:, 3]]) ], axis=1))
        hv_seed_time = hv_seed_time/tot_seed_time
        # print("\thv_seed_time:", hv_seed_time)

        hv_influence_seed = metric_influence_seed(F[:, :2])
        hv_influence_seed = hv_influence_seed/tot_influence_seed
        # print("\thv_influence_seed:", hv_influence_seed)

        hv_influence_seedSize_communities = metric_influence_seedSize_communities(F[:, :3])
        hv_influence_seedSize_communities = hv_influence_seedSize_communities/tot_influence_seedSize_communities
        # print("\thv_influence_seedSize_communities:", hv_influence_seedSize_communities)

        hv_influence_seedSize_time = metric_influence_seedSize_time(np.concatenate([F[:, :2] , np.transpose( [F[:, 3]]) ], axis=1))
        hv_influence_seedSize_time = hv_influence_seedSize_time/tot_influence_seedSize_time
        # print("\thv_influence_seedSize_time:", hv_influence_seedSize_time)

        # TODO: compute and save all other HV

        args["hypervolume"].append([hv_influence_seed, hv_influence_seedSize_time, hv_influence_seedSize_communities,
        hv_influence_communities, hv_seed_communities, hv_influence_time, hv_seed_time,  hv_influence_seedSize_communities_time ])
    
    elif args["elements_objective_function"] == "influence_time": 

        # TODO: save number of influenced

        hv_influence_time = metric_influence_time(F)
        hv_influence_time = hv_influence_time/tot_influence_time
        args["hypervolume"].append([hv_influence_time])

    
    elif args["elements_objective_function"] == "influence_communities": 

        hv_influence_communities = metric_influence_communities(F)
        hv_influence_communities = hv_influence_communities/tot_influence_communities
        args["hypervolume"].append([hv_influence_communities])

    elif args["elements_objective_function"] == "influence_communities_time": 

        hv_influence_time = metric_influence_time(np.concatenate([F[:, :1] , np.transpose( [F[:, 2]]) ], axis=1))
        hv_influence_time = hv_influence_time/tot_influence_time

        hv_influence_communities = metric_influence_communities(F[:, :2])
        hv_influence_communities = hv_influence_communities/tot_influence_communities

        args["hypervolume"].append([hv_influence_time, hv_influence_communities])


    # logging.info('Hypervolume at generation {0} : '.format(num_generations))   

    