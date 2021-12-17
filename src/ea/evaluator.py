from src.threadpool import ThreadPool
from src.spread.monte_carlo import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop

import inspyred
import threading
def nsga2_evaluator(candidates, args):
    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]
    communities = args["communities"]
    fitness_function = args["fitness_function"]
    fitness_function_kargs = args["fitness_function_kargs"]
    k = args["max_seed_nodes"]
    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 
    if n_threads == 1 :
        for index, A in enumerate(candidates) :

            A_set = set(A)

            if fitness_function != MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model, communities]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            influence_mean, influence_std, comm = fitness_function(*fitness_function_args, **fitness_function_kargs)
            #influence_mean, influence_std= fitness_function(*fitness_function_args, **fitness_function_kargs)

            fitness[index] = inspyred.ec.emo.Pareto([influence_mean, float(1.0 / len(A_set)), comm])
            #fitness[index] = inspyred.ec.emo.Pareto([influence_mean, float(1.0 / len(A_set))])

    else :
        thread_pool = ThreadPool(n_threads)

        # create thread lock, to be used for concurrency

        thread_lock = threading.Lock()

        # create list of tasks for the thread pool, using the threaded evaluation function
        tasks = []
        for index, A in enumerate(candidates) :
            A_set = set(A)
            if fitness_function != MonteCarlo_simulation_max_hop:
                fitness_function_args = [G, A_set, p, no_simulations, model, communities]
            else:
                max_hop = args["max_hop"]
                fitness_function_args = [G, A_set, p, no_simulations, model, max_hop]

            tasks.append((fitness_function, fitness_function_args, fitness_function_kargs, fitness, A_set, index,k, thread_lock))

        thread_pool.map(nsga2_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    return fitness


def nsga2_evaluator_threaded(fitness_function, fitness_function_args, fitness_function_kargs, fitness_values, A_set, index, k,thread_lock, thread_id) :

    influence_mean, influence_std, comm = fitness_function(*fitness_function_args, **fitness_function_kargs)

    # lock data structure before writing in it
    thread_lock.acquire()
   
    fitness_values[index] = inspyred.ec.emo.Pareto([influence_mean, k-len(A_set), comm])
    #fitness_values[index] = inspyred.ec.emo.Pareto([influence_mean, len(A_set), comm])

    thread_lock.release()

    return 