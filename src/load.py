import logging
import networkx as nx
import numpy as np
""" Graph loading """

def read_graph(filename, nodetype=int):

	graph_class = nx.DiGraph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	msg = ' '.join(["Read from file", filename, "the directed graph\n", nx.classes.function.info(G)])
	logging.info(msg)
	# filename = "RANDOM_Graph_SBM"
	# sizes = [60,70,200,100,150]

	# N=len(sizes)
	# #a = np.random.rand(N, N)
	# a = np.random.uniform(0.0005, 0.005, N)
	# m = np.tril(a) + np.tril(a, -1).T
	# m = m.tolist()
	# x = 0

	# ## to change the probability of edges in a community
	# for i in range(len(m)):
	# 	new = m[i]
	# 	new[x] = np.random.uniform(0.1, 0.2, 1)
	# 	m[i]=new
	# 	x= x+1


	# G = nx.stochastic_block_model(sizes, m, seed=0)
	return G

if __name__ == '__main__':

	logger = logging.getLogger('')
	logger.setLevel(logging.INFO)
	#read_graph("graphs/Email_URV.txt")
