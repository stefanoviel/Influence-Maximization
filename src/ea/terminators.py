
def generation_termination(population, num_generations, num_evaluations, args):
	"""
	generation termination function
	key args argument: generations_budget
	:param population:
	:param num_generations:
	:param num_evaluations:
	:param args:
	:return:
	"""
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
    ##???
    current_best = max(population).fitness[0]
    print(current_best, previous_best)
    if previous_best is None or previous_best != current_best:
        args['previous_best'] = current_best
        args['generation_count'] = 0
        return False
    else:
        if args['generation_count'] >= max_generations:
            return True
        else:
            args['generation_count'] += 1

            return False