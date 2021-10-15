import inspyred

@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2_super_operator(random, candidate1, candidate2, args) :
      
    children = []

    # uniform choice of operator
    randomChoice = random.randint(0,3)
    #randomChoice = 0

    if randomChoice == 0 :
        children = nsga2_crossover(random, list(candidate1), list(candidate2), args)
    elif randomChoice == 1 :
        children.append( ea_alteration_mutation(random, list(candidate1), args) )
    elif randomChoice == 2 :
        children.append(nsga2_insertion_mutation(random, list(candidate1), args) )
    elif randomChoice == 3 :
        children.append(nsga2_removal_mutation(random, list(candidate1), args) )

    # purge the children from "None" and empty arrays
    children = [c for c in children if c is not None and len(c) > 0]
    
    # this should probably be commented or sent to logging
    #for c in children : logging.debug("randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)) )
    
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