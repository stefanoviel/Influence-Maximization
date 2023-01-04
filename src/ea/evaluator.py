from src.threadpool import ThreadPool
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop

import inspyred
import threading
def nsga2_evaluator(candidates, args):
    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]
    fitness_function = args["fitness_function"] 
    fitness_function_kargs = args["fitness_function_kargs"]
    k = args["max_seed_nodes"]
    no_obj =  args["no_obj"]
    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 

    #calculate Time (Activation Attempts) for every individual in the population 
    time_gen = [None] * len(candidates)
    if n_threads == 1 :
        for index, A in enumerate(candidates) :

            A_set = set(A)

            if fitness_function != MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            if no_obj == 3:
                influence_mean, _,comm, time = fitness_function(*fitness_function_args, **fitness_function_kargs)
                time_gen[index] = time
                if args["elements_objective_function"] == "influence_seedSize_time": 
                    fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), 1/time])
                elif args["elements_objective_function"] == "influence_seedSize_communities": 
                    fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), comm])
                elif args["elements_objective_function"] == "influence_seedSize_communities_time": 
                    fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), comm, 1/time])
                else: 
                    raise Exception("Elements of the objective function aren't specified correctly")


            elif no_obj == 2:
                influence_mean, _, time = fitness_function(*fitness_function_args, **fitness_function_kargs)
                time_gen[index] = time
                if args["elements_objective_function"] == "influence_seedSize": 
                    fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100)])
                else: 
                    raise Exception("Elements of the objective function aren't specified correctly")

    else :
        thread_pool = ThreadPool(n_threads)

        # create thread lock, to be used for concurrency

        thread_lock = threading.Lock()

        # create list of tasks for the thread pool, using the threaded evaluation function
        tasks = []
        for index, A in enumerate(candidates):
            A_set = set(A)
            if fitness_function != MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            tasks.append((fitness_function, fitness_function_args, fitness_function_kargs, fitness, A_set, index,k, G,no_obj,time_gen,thread_lock, args))

        thread_pool.map(nsga2_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    args["time"].append(time_gen)
    return fitness


def nsga2_evaluator_threaded(fitness_function, fitness_function_args, fitness_function_kargs, fitness_values, A_set, index, k,G, no_obj,time_gen_values,thread_lock, args, thread_id) :

    influence_mean, _, comm, time = fitness_function(*fitness_function_args, **fitness_function_kargs)

    # lock data structure before writing in it
    thread_lock.acquire()
    # if no_obj == 3:
    #     print("INSIDE")
    #     fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), comm])
    #     time_gen_values[index] = time
    # elif no_obj == 2:
    #     fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100)])

    if no_obj == 3:
        if args["elements_objective_function"] == "influence_seedSize_time": 
            fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), time])
        elif args["elements_objective_function"] == "influence_seedSize_communities": 
            fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), comm])
            # print("INSIDE:", fitness_values[index])
        elif args["elements_objective_function"] == "influence_seedSize_communities_time": 
            fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100), comm, time])
        else: 
            raise Exception("Elements of the objective function aren't specified correctly")

    elif no_obj == 2:
        influence_mean, _, time = fitness_function(*fitness_function_args, **fitness_function_kargs)
        time_gen_values[index] = time
        if args["elements_objective_function"] == "influence_seedSize": 
            fitness_values[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) * 100, (((k-len(A_set)) / G.number_of_nodes()) * 100)])
        else: 
            raise Exception("Elements of the objective function aren't specified correctly")

    thread_lock.release()

    return 