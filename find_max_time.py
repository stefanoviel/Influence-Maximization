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


def read_graph(filename, nodetype=int):

	graph_class = nx.Graph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	return G

all_max_times = {}


def process_graph(graph):
    G = read_graph(os.path.join('graphs_downscaled', graph))
    print(graph, len(G), len(G)* 0.025)

    args = {}
    match = re.search(r'\d+', graph)
    if match: 
        s = match.group()
        args['s'] = s
        args['downscaled'] = True
    else: 
        args['downscaled'] = False

    args['graph'] = graph.replace('_' + s + '.txt', '')

    if int(s) == 4: 
        times = []
        random_nodes = sample(list(G.nodes()), round(len(G)* 0.025))
        try: 
            communities = get_communities(args)
            time = MonteCarlo_simulation_max_time(G, random_nodes, 0.05, 50000, 'IC', communities=communities)
            times.append(time)
        except FileNotFoundError as fnf: 
            pass
    
        print(max(times))
        all_max_times[graph.replace('.txt', '')] = [max(times)]


def worker():
    while True:
        graph = q.get()
        process_graph(graph)
        q.task_done()

# Create a queue to hold the graphs
q = Queue()

# Create and start worker threads
for i in range(6):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

# Add the graphs to the queue
for graph in os.listdir('graphs_downscaled'):
    q.put(graph)

# Wait for all tasks to be completed
q.join()

df = pd.DataFrame(all_max_times)
print(df)
df.to_csv('networks_max_times.csv')

    
