
from inspyred.ec.analysis import hypervolume
from pymoo.factory import get_performance_indicator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def generation_termination(population, num_generations, num_evaluations, args):
    if num_generations == args["generations_budget"]:
        x = [x for x in range(1,len(args["hypervolume"])+1)]
        import matplotlib.pyplot as plt

        plt.plot(x, args["hypervolume"], color='b')
        plt.xlabel('Generations')
        plt.ylabel('Hypervolume')
        plt.savefig(args["population_file"]+'.png')
        plt.cla()
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
        print(max_generations,args["generation_count"])
    except:
        pass

    pop = [list(x.fitness) for x in args["_ec"].archive] 
    F =  np.array(pop)

    tot = args["graph"].number_of_nodes() * 1 * len(args["communities"])
    ref_point = np.array([args["graph"].number_of_nodes(),1, len(args["communities"])])
    
    hv = get_performance_indicator("hv", ref_point=ref_point)
    hv =hv.do(F)
    current_best = round(1- hv/tot,3)
    print(hv, hv/tot)
    print("Hypervolume {0}-{1} Generations {2}".format(current_best,previous_best, num_generations))
    print('Len of archive {0}'.format(len(pop)))
    args["hypervolume"].append(current_best)

    # one = []
    # two = []
    # three = []
    # for item in pop:
    #     one.append(item[0])
    #     two.append(item[1])
    #     three.append(item[2])
    # df = pd.DataFrame()
    # df["influence"] = one
    # df["nodes"] = two
    # df["comm"] = three
    if previous_best is None or previous_best < current_best:
        args['previous_best'] = current_best
        args['generation_count'] = 0
        #df.to_csv('best.csv', index=False)
        return False
    else:
        if args['generation_count'] >= max_generations:
            x = [x for x in range(1,len(args["hypervolume"])+1)]

            plt.plot(x, args["hypervolume"])
            plt.xlabel('Generations')
            plt.ylabel('Hypervolume')
            plt.savefig(args["population_file"]+'.png')
            plt.cla()
            return True
        else:
            args['generation_count'] += 1
            #df.to_csv('worse.csv', index=False)
            return False
