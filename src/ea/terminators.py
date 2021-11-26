
#from inspyred.ec.analysis import hypervolume
from pymoo.factory import get_performance_indicator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def generation_termination(population, num_generations, num_evaluations, args):
    if num_generations == args["generations_budget"]:
        x = [x for x in range(1,len(args["hypervolume"])+1)]
        import matplotlib.pyplot as plt
        plt.cla()
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
        print('Max Generations without improvements{0}, now {1}'.format(max_generations,args["generation_count"]))
    except:
        pass

    arch = [list(x.fitness) for x in args["_ec"].archive] 
    import copy
    original_arch = copy.deepcopy(arch)
    for i in range(len(arch)):
        for j in range(len(arch[i])):
            arch[i][j] = -  arch[i][j]
    F =  np.array(arch)

    tot = args["graph"].number_of_nodes() * 1 * len(args["communities"])

    t = 1/args["graph"].number_of_nodes()

    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([-1,-t,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)
    hv = metric.do(F)
    current_best = hv/tot

    print("Hypervolume {0}-{1} Generations {2}".format(current_best,previous_best, num_generations))
    args["hypervolume"].append(current_best)
    one = []
    two = []
    three = []
    for item in original_arch:
        one.append(item[0])
        two.append(1/item[1])
        three.append(item[2])
    df = pd.DataFrame()
    df["influence"] = one
    df["nodes"] = two
    df["comm"] = three

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(two,three,one, color='red', alpha=1,linewidth=0)
    ax.set_xlabel("No nodes")

    ax.set_zlabel("Influence")

    ax.set_ylabel("Communities")

    ax.set_ylim([1,len(args["communities"])])
    #ax.set_zlim([1,550])
    ax.set_xlim([1,(args["max_seed_nodes"])])
    plt.savefig('PF/'+ str(num_generations)+'.png')
    #plt.show()

    if previous_best is None or current_best > previous_best:
        args['previous_best'] = current_best
        args['generation_count'] = 0
        return False
    else:
        if args['generation_count'] >= max_generations:
            plt.cla()
            x = [x for x in range(1,len(args["hypervolume"])+1)]
            plt.plot(x, args["hypervolume"])
            plt.xlabel('Generations')
            plt.ylabel('Hypervolume')
            plt.title('Generations {0}'.format(num_generations))
            plt.savefig(args["population_file"]+'.png')
            plt.cla()
            return True
        else:
            args['generation_count'] += 1
            return False
