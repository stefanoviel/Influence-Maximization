from networkx.generators import intersection
import numpy
import math
import random
import os
import pandas as pd
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
def LT_model(G, a, p, communities,random_generator,  max_time= math.inf):
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    threshold = {}
    l = np.random.uniform(low=0.0, high=1.0, size=G.number_of_nodes())
    degree_list = {}	
    activate = {}
    for i, node in enumerate(G.nodes()):
            threshold[node] = l[i]
            degree_list[node] = float(1/G.degree(node))
            activate[node] = len(set.intersection(set(G.neighbors(node)),set(A))) 
    time = 0
    while not converged and time <= max_time:
        nextB = set()
        S = []
        time += 1    			 	
        for n in B: 
            for m in set(G.neighbors(n)) - A - set(S):
                S.append(m)
                if activate[m] * degree_list[m] > threshold[m]:
                    nextB.add(m)
                    for t in set(G.neighbors(m)) - A:
                        activate[t] +=1
        B = set(nextB)
        if not B:
            converged = True
        A |= B 
    comm = 0
    for item in communities:
        intersection = set.intersection(set(item),set(A))
        if len(intersection) > 0:
            comm += 1
				    	
    return len(A), comm, time


	#This returns all the nodes in the network that have been activated/converted in the diffusion process

def IC_model(G, a, p,communities, random_generator, max_time= math.inf):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)  
	converged = False
	time = 0

	while not converged and time <= max_time:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		time = time+1  
		if not B:
			converged = True
		A |= B
	
	# if time >= max_time: 
	# 	print('exited because of max_time')
	# else: 
	# 	print('exited not because of time')

	comm = 0
	for item in communities:
		intersection = set.intersection(set(item),set(A))
		if len(intersection) > 0:
			comm += 1	
	return len(A), comm, time


def IC_model_influenced_nodes(G, a, p, random_generator, max_time= math.inf):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)  
	converged = False
	time = 0

	while not converged and time <= max_time:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		time = time+1  
		if not B:
			converged = True
		A |= B
	

	return A

def IC_model_community_seed(G, a, p, communities, random_generator, max_time= math.inf):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)  
	converged = False
	time = 0

	# compute communities on original seed set
	comm = 0
	for item in communities:
		intersection = set.intersection(set(item),set(A))
		if len(intersection) > 0:
			comm += 1

	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		time = time+1  
		if not B:
			converged = True
		A |= B
		
	return len(A), comm, time

def WC_model(G, a, communities,random_generator, max_time= math.inf):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)                      # B: the set of nodes activated in the last completed iteration
	converged = False
	time = 0                  # B: the set of nodes activated in the last completed iteration

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree
	
	while not converged and time <= max_time:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random() # in the range [0.0, 1.0)
				p = 1.0/my_degree_function(m) 
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		time += 1
		if not B:
			converged = True
		A |= B
	comm = 0
	for item in communities:
		intersection = set.intersection(set(item),set(A))
		if len(intersection) > 0:
			comm += 1	
	return len(A), comm, time



def MonteCarlo_simulation_time(G, A, p, no_simulations, model, communities, max_time, random_generator=None):

	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []
	comm_list = []
	times = []

	if model == 'WC':
		for i in range(no_simulations):
			res, comm, time = WC_model(G, A, communities=communities,random_generator=random_generator, max_time=max_time)
			comm_list.append(comm)
			results.append(res)
			times.append(time)
	elif model == 'IC':
		for i in range(no_simulations):
			res, comm ,time= IC_model(G, A, p, communities=communities, random_generator=random_generator, max_time=max_time)
			comm_list.append(comm)
			results.append(res)
			times.append(time)
	elif model == 'LT':
		for i in range(no_simulations):
			res, comm , time= LT_model(G, A, p, communities=communities,random_generator=random_generator,  max_time=max_time)
			comm_list.append(comm)
			results.append(res)
			times.append(time)


	set_com_time = {'set': A, 'communities':comm, 'time':(time)}

	return (numpy.mean(results), numpy.std(results), int(numpy.mean(comm_list)), int(numpy.mean(times)), set_com_time)


def MonteCarlo_simulation_max_time(G, A, p, no_simulations, model, communities, random_generator=None):
	"""Same as function before but doesn't take max_time and only return the max time (not the average as befor )"""

	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []
	comm_list = []
	times = []

	if model == 'WC':
		for i in range(no_simulations):
			res, comm, time = WC_model(G, A,  communities=communities,random_generator=random_generator)
			comm_list.append(comm)
			results.append(res)
			times.append(time)
	elif model == 'IC':
		for i in range(no_simulations):
			res, comm ,time= IC_model(G, A, p,  communities=communities, random_generator=random_generator)
			comm_list.append(comm)
			results.append(res)
			times.append(time)
	elif model == 'LT':
		for i in range(no_simulations):
			res, comm , time= LT_model(G, A, p,  communities=communities,random_generator=random_generator)
			comm_list.append(comm)
			results.append(res)
			times.append(time)

	return max(times)
