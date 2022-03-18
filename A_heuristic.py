from src.load import read_graph
from src_OLD.heuristics.SNInflMaxHeuristics_networkx import *
G = read_graph('graphs/fb_politician.txt')
import time
start_time = time.time()
t = CELF(int(G.number_of_nodes() * 0.025), G, 0, 100, 'WC')

print(time.time()-start_time)
print(int(G.number_of_nodes() * 0.025))
