import os
import threading
from queue import Queue
import logging
import pandas as pd
import numpy as np
import re
from random import sample
from src.spread.monte_carlo_3_obj import MonteCarlo_simulation_max_time 
from influence_maximization import get_communities
import networkx as nx
import math
import numpy as np


considered_graphs = ['deezerEU_4.txt', 'facebook_combined_4.txt', 'fb_politician_4.txt', 'pgp_4.txt', 'cora.txt']


def read_graph(filename, nodetype=int):

	graph_class = nx.Graph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	return G

all_max_times = {}


def process_graph(base_path, graph):
    G = read_graph(os.path.join(base_path, graph))
    print(graph, len(G), len(G)* 0.025)

    args = {}
    match = re.search(r'\d+', graph)
    if match: 
        s = match.group()
        args['s'] = s
        args['downscaled'] = True
        args['graph'] = graph.replace('_' + s + '.txt', '')
    else: 
        args['downscaled'] = False
        args['graph'] = graph.replace('.txt', '')

    if graph in considered_graphs: 
        times = []
        random_nodes = sample(list(G.nodes()), round(len(G)* 0.025))
        try: 
            communities = get_communities(args)
            time = MonteCarlo_simulation_max_time(G, random_nodes, 0.05, 50000, 'IC', communities=communities)
            times.append(time)
        except FileNotFoundError as fnf: 
            pass
    
        all_max_times[graph.replace('.txt', '')] = [max(times)]


def worker():
    while True:
        base_path, graph = q.get()
        process_graph(base_path, graph)
        q.task_done()

# Create a queue to hold the graphs
q = Queue()

# Create and start worker threads
for i in range(6):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

# Add the graphs to the queue
paths = [('graphs_downscaled', p) for p in os.listdir('graphs_downscaled') ]
paths =  paths + [('graphs', p) for p in os.listdir('graphs') ]

for base_path, graph in paths:
    q.put((base_path, graph))
    # process_graph(os.path.join(base_path, graph))

# Wait for all tasks to be completed
q.join()

df = pd.DataFrame(all_max_times)

df.to_csv('networks_max_times.csv')

    
