import inspyred
@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def nsga2_generator(random, args) :

    min_seed_nodes = args["min_seed_nodes"]
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    #logging.debug("Min seed set size: %d; Max seed set size: %d" % (min_seed_nodes, max_seed_nodes))

    # extract random number in 1,max_seed_nodes
    individual_size = random.randint(min_seed_nodes, max_seed_nodes)
    individual = [0] * individual_size
    #logging.debug( "Creating individual of size %d, with genes ranging from %d to %d" % (individual_size, nodes[0], nodes[-1]) )
    
    for i in range(0, individual_size) : individual[i] = nodes[ random.randint(0, len(nodes)-1) ]
    #logging.debug(individual)

    return list(set(individual))