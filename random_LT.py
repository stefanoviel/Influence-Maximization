from re import M
from src.load import read_graph
import time
from networkx.generators import intersection
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

def LT_model_slow(G, a, p, communities,random_generator):
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    threshold = {}
    l = np.random.uniform(low=0.0, high=1.0, size=G.number_of_nodes())
    import time
    for i, node in enumerate(G.nodes()):
            threshold[node] = l[i]
    #s = time.time()
    time_ = 0
    while not converged:
        nextB = set()
        for n in B: 
            for m in set(G.neighbors(n)) - A:
                time_ += 1    			
                total_weight = 0
                weight = float(1/G.degree(m))
                for each in G.neighbors(m):
                    if each in A:
                        total_weight =  total_weight + weight			
                if total_weight > threshold[m]:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B 
    #e = time.time() - s  
    #print(e)       
    comm = 0
    
    for item in communities:
        intersection = set.intersection(set(item),set(A))
        if len(intersection) > 0:
            comm += 1
				    	
    return len(A), comm, time_

def LT_model(G, a, p, communities, degree, join, random_generator):
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    threshold = {}
    l = np.random.uniform(low=0.0, high=1.0, size=G.number_of_nodes())
    degree_list = {}	
    activate = {}
    import time
    for i, node in enumerate(G.nodes()):
            threshold[node] = l[i]
            degree_list[node] = degree[node]
            activate[node] = join[node]
    #s = time.time()
    time_ = 0
    while not converged:
        nextB = set()
        S = []
        for n in B: 
            for m in (set(G.neighbors(n)) - A - set(S)):
                S.append(m)
                time_ += 1    			 	
                if activate[m] * degree_list[m] > threshold[m]:
                    nextB.add(m)
        for node in nextB:
            for t in set(G.neighbors(node)) - A:
                activate[t] +=1
                #S.append(t)
        B = set(nextB)
        if not B:
            converged = True
        A |= B 
    #e = time.time() - s  
    #print(e)
    comm = 0
    for item in communities:
        intersection = set.intersection(set(item),set(A))
        if len(intersection) > 0:
            comm += 1
				    	
    return len(A), comm, time_

	#This returns all the nodes in the network that have been activated/converted in the diffusion process

def IC_model(G, a, p, communities, random_generator):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)  
	converged = False
	time = 0

	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				time = time+1  
				if prob <= p:
					nextB.add(m)
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


def WC_model(G, a, communities,random_generator):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)                      # B: the set of nodes activated in the last completed iteration
	converged = False

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree
	time = 0
	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random() # in the range [0.0, 1.0)
				p = 1.0/my_degree_function(m) 
				time +=1
				if prob <= p:
					nextB.add(m)
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



def WC_model_2(G, a, communities, degree,random_generator):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)                      # B: the set of nodes activated in the last completed iteration
	converged = False

	time = 0
	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random() # in the range [0.0, 1.0)
				p = degree[m]
				time +=1
				if prob <= p:
					nextB.add(m)
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

""" Evaluates a given seed set A, simulated "no_simulations" times.
	Returns a tuple: (the mean, the stdev).
"""
def MonteCarlo_simulation(G, A, p, no_simulations, model, communities, degree, join ,random_generator=None):
	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []
	times = []
	comm_list = []
	if model == 'WC':
		for i in range(no_simulations):
			res, comm, time  = WC_model(G, A, communities,random_generator=random_generator)
			times.append(time)
			results.append(res)
			comm_list.append(comm)
	elif model == 'IC':
		for i in range(no_simulations):
			res, comm, time  = IC_model(G, A, p, communities, random_generator=random_generator)
			times.append(time)
			results.append(res)
			comm_list.append(comm)
	elif model == 'LT':
		for i in range(no_simulations):
			res, comm, time = LT_model(G, A, p,communities, degree, join, random_generator=random_generator)
			times.append(time)
			results.append(res)
			comm_list.append(comm)
	elif model == 'WC2':
		for i in range(no_simulations):
			res, comm, time = WC_model_2(G, A,communities, degree, random_generator=random_generator)
			times.append(time)
			results.append(res)
			comm_list.append(comm)
			results.append(res)
	elif model == 'LT2':
		for i in range(no_simulations):
			res, comm, time = LT_model_slow(G, A, p,communities, random_generator=random_generator)
			times.append(time)
			results.append(res)
			comm_list.append(comm)

	return (numpy.mean(results), numpy.std(results), int(numpy.mean(comm_list)),sum(times))

G = read_graph('graphs/deezerEU.txt')

N = 5
MAX = 100
no_simulations = 100

m_values = G.number_of_nodes()
seed_sets = []

for i in range(N):
    import random
    k = random.randint(1,MAX)
    seed_sets.append(random.sample(range(1, m_values), k))

import pandas as pd
communities =[]
df = pd.read_csv('comm_ground_truth/deezerEU.csv',sep=",")
print(df)
groups = df.groupby('comm')['node'].apply(list)
print(groups)
df = groups.reset_index(name='nodes')
communities = df["nodes"].to_list()

p = 0.2
model = ["LT","LT2"]
import time



for m in model:
    start = time.time()
    for id, item in enumerate(seed_sets):
        A = set(item)
        degree = {}
        join = {}
        for node in G:
            degree[node] = 1/G.degree(node)
            join[node] = len(set.intersection(set(G.neighbors(node)),set(A)))
        influence,std, comm, t = MonteCarlo_simulation(G, A, p, no_simulations, m, communities, degree, join, random_generator=None)
        print('i='+str(id),len(A),influence, comm, std)

    exec_time = time.time() - start   
    print(m, exec_time, t)   
