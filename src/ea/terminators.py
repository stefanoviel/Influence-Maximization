import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.factory import get_performance_indicator
#local libraries
from src.ea.observer import time_observer


def generation_termination(population, num_generations, num_evaluations, args):
    """
    Return true when reached the maximum number of generations.
    Save into format .csv the Hypervolume and Time lists reached/used the whole evolution process. 
    """
    if num_generations == args["generations_budget"]:
        x = [x for x in range(1,len(args["hypervolume"])+1)]
        df = pd.DataFrame()
        df["generation"] = x

        if args["elements_objective_function"] == "influence_seedSize": 
            df["hv_influence_seed"] = np.array(args["hypervolume"])[:, 0]
            
        elif args["elements_objective_function"] == "influence_seedSize_time": 
            df["hv_influence_seedSize_time"] = np.array(args["hypervolume"])[:, 0]
            df["hv_influence_seed"] = np.array(args["hypervolume"])[:, 1]
            df["hv_influence_time"] = np.array(args["hypervolume"])[:, 2]
            df["hv_seed_time"] = np.array(args["hypervolume"])[:, 3]

        elif args["elements_objective_function"] == "influence_seedSize_communities": 
            df["hv_influence_seedSize_communities"] = np.array(args["hypervolume"])[:, 0]
            df["hv_influence_seed"] = np.array(args["hypervolume"])[:, 1]
            df["hv_influence_communities"] = np.array(args["hypervolume"])[:, 2]
            df["hv_seed_communities"] = np.array(args["hypervolume"])[:, 3]

        elif args["elements_objective_function"] == "influence_seedSize_communities_time": 
            df["hv_influence_seedSize_communities_time"] = np.array(args["hypervolume"])[:, 0]
            df["hv_influence_communities"] = np.array(args["hypervolume"])[:, 1]
            df["hv_seed_communities"] = np.array(args["hypervolume"])[:, 2]
            df["hv_influence_time"] = np.array(args["hypervolume"])[:, 3]
            df["hv_seed_time"] = np.array(args["hypervolume"])[:, 4]
            df["hv_influence_seed"] = np.array(args["hypervolume"])[:, 5]
          
        df.to_csv(args["population_file"] +"_hv_.csv", sep=",",index=False)
        time_observer(population, num_generations, num_evaluations, args)

    return num_generations == args["generations_budget"]

def no_improvement_termination(population, num_generations, num_evaluations, args):
    """Return True if the best fitness does not change for a number of generations.
    
    This function keeps track of the current best fitness and compares it to
    the best fitness in previous generations. Whenever those values are the 
    same, it begins a generation count. If that count exceeds a specified 
    number, the terminator returns True.
    
    .. Arguments:
       population -- the population of Individuals
       num_generations -- the number of elapsed generations
       num_evaluations -- the number of candidate solution evaluations
       args -- a dictionary of keyword arguments
    
    Optional keyword arguments in args:
    
    - *max_generations* -- the number of generations allowed for no change in fitness (default 10)
    
    """
    max_generations = args.setdefault('max_generations', 10)
    previous_best = args.setdefault('previous_best', None)
    try:
        previous_best = args["hypervolume"][-2]
    except:
        pass
    
    current_best = args["hypervolume"][-1]

    if previous_best is None or current_best > previous_best:
        args['previous_best'] = current_best
        args['generation_count'] = 0
        return False
    else:
        if args['generation_count'] >= max_generations:
            x = [x for x in range(1,len(args["hypervolume"])+1)]
            df = pd.DataFrame()
            df["generation"] = x
            df["hv"] = args["hypervolume"]
            df.to_csv(args["population_file"] +"_hv_.csv", sep=",",index=False)
            time_observer(population, num_generations, num_evaluations, args)
            return True
        else:
            args['generation_count'] += 1
            return False
