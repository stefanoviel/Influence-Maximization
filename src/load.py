import logging
import networkx as nx
import numpy as np
""" Graph loading """

def read_graph(filename, nodetype=int):

	graph_class = nx.Graph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	msg = ' '.join(["Read from file", filename, "the directed graph\n", nx.classes.function.info(G)])
	return G

if __name__ == '__main__':

	logger = logging.getLogger('')
	logger.setLevel(logging.INFO)
