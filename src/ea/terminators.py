
#from inspyred.ec.analysis import hypervolume
from pymoo.factory import get_performance_indicator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generation_termination(population, num_generations, num_evaluations, args):
    if num_generations == args["generations_budget"]:
        x = [x for x in range(1,len(args["hypervolume"])+1)]
        #plt.cla()
        plt.plot(x, args["hypervolume"], color='b')
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.savefig(args["population_file"]+'.png')
        plt.cla()
        df = pd.DataFrame()
        df["generation"] = x
        df["hv"] = args["hypervolume"]
        df["influence_k"] = args["hv_influence_k"]
        df["influence_comm"] = args["hv_influence_comm"]
        df["k_comm"] = args["hv_k_comm"]
        df.to_csv(args["population_file"] +"_hv_.csv", sep=",",index=False)
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
        previous_best = args["hypervolume"][-1]

    except:
        pass


    arch = [list(x.fitness) for x in args["_ec"].archive] 
    import copy
    original_arch = copy.deepcopy(arch)
    for i in range(len(arch)):
        for j in range(len(arch[i])):
                arch[i][j] = -float(arch[i][j])
    F =  np.array(arch)

    t = (1/args["max_seed_nodes"]) 
    print(t, (1-t))
    tot = args["graph"].number_of_nodes() * (1 - (1-t)) * len(args["communities"])

    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([-1,-t,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)
    hv = metric.do(F)
    current_best = hv/tot
    print("Hypervolume {0}-{1} Generations {2}".format(current_best,hv, num_generations))
    args["hypervolume"].append(current_best)
    # HV INFLUENCE - K
    arch_2 = []
    for i in range(len(original_arch)):
        obj = []
        for j in range(len(original_arch[i])):    
            if j != 2:
                obj.append(-float(original_arch[i][j]))            
        arch_2.append(obj)
    metric = Hypervolume(ref_point= np.array([-1,-t]),
                        norm_ref_point=False,
                        zero_to_one=False)
    F1 = np.array(arch_2)
    tot_1 =args["graph"].number_of_nodes() * (1 - (1-t))
    hv_1 = metric.do(F1)
    b = hv_1/tot_1
    args["hv_influence_k"].append(b)
    print('INFLUENCE-K {0}'.format(b))
    
    # HV INFLUENCE - COMM
    arch_2 = []
    for i in range(len(original_arch)):
        obj = []
        for j in range(len(original_arch[i])):
            if j != 1:
                obj.append(-float(original_arch[i][j]))            
        arch_2.append(obj)
    metric = Hypervolume(ref_point= np.array([-1,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)
    F1 = np.array(arch_2)
    tot_1 =args["graph"].number_of_nodes() * len(args["communities"])  
    hv_1 = metric.do(F1)
    b = hv_1/tot_1
    args["hv_influence_comm"].append(b)
    print('INFLUENCE-COMM {0}'.format(b))

    #HV K - COMM
    arch_2 = []
    for i in range(len(original_arch)):
        obj = []
        for j in range(len(original_arch[i])):
            if j != 0:
                obj.append(-float(original_arch[i][j]))            
        arch_2.append(obj)
    metric = Hypervolume(ref_point= np.array([-t,-1]),
                        norm_ref_point=False,
                        zero_to_one=False)
    F1 = np.array(arch_2)
    tot_1 = (1-(1-t)) * len(args["communities"]) 
    hv_1 = metric.do(F1)
    b = hv_1/tot_1
    args["hv_k_comm"].append(b)
    print('K-COMM {0}'.format(b))


    if previous_best != None:
        print('Current Best - Previous Best {0}'.format((current_best-previous_best)))
    # one = []
    # two = []
    # three = []
    # for item in original_arch:
    #     one.append(item[0])
    #     two.append(1/item[1])
    #     three.append(item[2])
    # df = pd.DataFrame()
    # df["influence"] = one
    # df["nodes"] = two
    # df["comm"] = three

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(two,three,one, color='red', alpha=1,linewidth=0)
    # ax.set_xlabel("No nodes")

    # ax.set_zlabel("Influence")

    # ax.set_ylabel("Communities")

    # ax.set_ylim([1,len(args["communities"])])
    #ax.set_zlim([1,550])
    #ax.set_xlim([1,(args["max_seed_nodes"])])
    #plt.title('Generations {0}'.format(num_generations))
    #plt.savefig('PF/'+ str(num_generations)+'.png')
    #plt.close()
    #plt.show()

    if previous_best is None or current_best > previous_best:
        args['previous_best'] = current_best
        args['generation_count'] = 0
        return False
    else:
        if args['generation_count'] >= max_generations:
            x = [x for x in range(1,len(args["hypervolume"])+1)]
            plt.plot(x, args["hypervolume"])
            plt.xlabel('Generation')
            plt.ylabel('Hypervolume')
            plt.savefig(args["population_file"]+'.png')
            plt.cla()
            df = pd.DataFrame()
            df["generation"] = x
            df["hv"] = args["hypervolume"]
            df["influence_k"] = args["hv_influence_k"]
            df["influence_comm"] = args["hv_influence_comm"]
            df["k_comm"] = args["hv_k_comm"]
            df.to_csv(args["population_file"] +"_hv_.csv", sep=",",index=False)
            return True
        else:
            args['generation_count'] += 1
            return False
