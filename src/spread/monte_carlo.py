import numpy
import random
import networkx as nx
import numpy as np
""" Spread models """

""" Simulation of spread for Independent Cascade (IC) and Weighted Cascade (WC). 
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""

''''
Added time inside the cycle of the various models of propagation with the purpose to keep track of how much time it takes the propagation to converge to the optimal solution.
'''

## to re-code better
def LT_model(G, a, p, communities,random_generator):
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    threshold = {}
    l = np.random.uniform(low=0.0, high=1.0, size=G.number_of_nodes())
    for i, node in enumerate(G.nodes()):
            threshold[node] = l[i]
    while not converged:
        nextB = set()
        for n in B: 
            for m in set(G.neighbors(n)) - A:
                total_weight = 0
                for each in G.neighbors(m):
                    if each in A:
                        prob = random_generator.random()  # in the range [0.0, 1.0)
                        if prob <= p:
                            total_weight =  total_weight + prob
                if total_weight > threshold[m]:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B
    comm = 0
    reach = {}
    for i in range(len(communities)):
	    reach[i] = False
    
    for item in A:
	    for key in range(len(communities)):
		    if reach[key] == False:
			    if item in communities[key]:
				    reach[key] = True
				    comm = comm + 1
	    if comm == len(communities):
		    break
				    	
    return len(A), comm


	#This returns all the nodes in the network that have been activated/converted in the diffusion process

def IC_model(G, a, p, communities, random_generator):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)  
	F = set(a)  

	converged = False
	comm = 0
	#time = 0

	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		#time = time+1  
		if not B:
			converged = True
		A |= B
	reach = {}
	for i in range(len(communities)):
		reach[i] = False
    
	for item in A:
		for key in range(len(communities)):
			if reach[key] == False:
				if item in communities[key]:
					reach[key] = True
					comm = comm + 1
		if comm == len(communities):
			break
	return len(A), comm

	#return len(A), comm, time

def WC_model(G, a, communities,random_generator):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)                      # B: the set of nodes activated in the last completed iteration
	converged = False

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree
	
	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random() # in the range [0.0, 1.0)
				p = 1.0/my_degree_function(m)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
	reach = {}
	comm = 0
	for i in range(len(communities)):
		reach[i] = False
    
	for item in A:
		for key in range(len(communities)):
			if reach[key] == False:
				if item in communities[key]:
					reach[key] = True
					comm = comm + 1
		if comm == len(communities):
			break
				    	
	return len(A), comm 
				    	
def IC_model_max_hop(G, a, p, max_hop, random_generator):  # a: the set of initial active nodes
	# p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False
	time = 0
	total_max_hop = max_hop
	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:  # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random()  # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1
		time = time + 1
	
	
	if converged != True:
		time = total_max_hop

	return len(A) , time


def WC_model_max_hop(G, a, max_hop, random_generator):  # a: the set of initial active nodes
	# each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree

	time = 0
	total_max_hop = max_hop

	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random()  # in the range [0.0, 1.0)
				p = 1.0 / my_degree_function(m)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1
		time += 1
	
	if converged != True:
		time = total_max_hop
	
	return len(A), time

""" Evaluates a given seed set A, simulated "no_simulations" times.
	Returns a tuple: (the mean, the stdev).
"""
def MonteCarlo_simulation(G, A, p, no_simulations, model, communities, random_generator=None):
	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []
	times = []
	tt = []
	if model == 'WC':
		for i in range(no_simulations):
			res, time = WC_model(G, A, communities,random_generator=random_generator)
			times.append(time)
			results.append(res)
	elif model == 'IC':
		for i in range(no_simulations):
			res, time = IC_model(G, A, p, communities, random_generator=random_generator)
			times.append(time)
			results.append(res)
			#tt.append(t)
	elif model == 'LT':
		for i in range(no_simulations):
			res, time = LT_model(G, A, p,communities,random_generator=random_generator)
			times.append(time)
			results.append(res)

	return (numpy.mean(results), numpy.std(results), int(numpy.mean(times)))

	#return (numpy.mean(results), numpy.std(results), int(numpy.mean(times)), numpy.mean(tt))

def MonteCarlo_simulation_max_hop(G, A, p, no_simulations, model, max_hop=5, random_generator=None):
	"""
	calculates approximated influence spread of a given seed set A, with
	information propagation limited to a maximum number of hops
	example: with max_hops = 2 only neighbours and neighbours of neighbours can be activated
	:param G: networkx input graph
	:param A: seed set
	:param p: probability of influence spread (IC model)
	:param no_simulations: number of spread function simulations
	:param model: propagation model
	:param max_hops: maximum number of hops
	:return:
	"""
	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []
	times = []
	if model == 'WC':
		for i in range(0,no_simulations):
			res, time = WC_model_max_hop(G, A, max_hop, random_generator)
			times.append(time)
			results.append(res)
			print('Time: {0} \nResults: {1} \n'.format(time,res))
	elif model == 'IC':
		for i in range(1):
			res, time = (IC_model_max_hop(G, A, p, max_hop, random_generator))
			times.append(time)
			results.append(res)
			print('Time: {0} \nResults: {1} \n'.format(time,res))
	elif model == 'LT':
		print("L original {0}".format(A))

		for i in range(no_simulations):
			res, time = LT_model(G, A, p,random_generator=random_generator)
			times.append(time)
			results.append(res)
			print('Simulation: {0} \nTime: {1} \nResults: {2} \n'.format(i,time,res))	

	return (numpy.mean(results), numpy.std(results), numpy.mean(times))


if __name__ == "__main__":

	G = nx.path_graph(100)
