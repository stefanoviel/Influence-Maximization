import sys
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from pymoo.indicators.hv import Hypervolume
from pymoo.factory import get_performance_indicator
from networkx.algorithms import degree_centrality, closeness_centrality, core_number, betweenness_centrality, katz_centrality_numpy, eigenvector_centrality_numpy
# local libraries
sys.path.insert(0, '')
from src.load import read_graph
from src.spread.monte_carlo import MonteCarlo_simulation

def read_arguments():
    """
	Parameters for the upscaling process process.
	"""
    parser = argparse.ArgumentParser(
        description='Upscalinf algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--graph', default='facebook_combined',
                        choices=['facebook_combined', 'fb_politician',
                                 'deezerEU', 'fb_org', 'fb-pages-public-figuree',
                                 'pgp', 'soc-gemsec', 'soc-brightkite'],
                        help='Graph name')

    # Upscaling Parameters
    parser.add_argument('--s', type=int, default=4,
                        help='Scaling factor')  
    parser.add_argument('--measure', default='betweenness',
                        choices=['two-hop','page_rank', 'degree_centrality',
                                'katz_centrality','betweenness', 'closeness', 
                                'eigenvector_centrality', 'core'])

    # Propagation parameters
    parser.add_argument('--p', type=float, default=0.01,
                        help='Probability of influence spread in the IC model.')
    parser.add_argument('--no_simulations', type=int, default=100,
                        help='Number of simulations for spread calculation'
                             ' when the Monte Carlo fitness function is used.')
    parser.add_argument('--model', default="WC", choices=['IC', 'WC'],
                        help='Influence propagation model.')
    parser.add_argument('--k', default=2.5, type=int,
                        help='Percentage of nodes as seed sets')
    args = parser.parse_args()
    args = vars(args)

    return args
#------------------------------------------------------------------------#


def n_neighbor(G, id, n_hop):
    node = [id]
    node_visited = set()
    neighbors= []
    
    while n_hop !=0:
        neighbors= []
        for node_id in node:
            node_visited.add(node_id)
            neighbors +=  [id for id in G.neighbors(node_id) if id not in node_visited]
        node = neighbors
        n_hop -=1
        
        if len(node) == 0 :
            return neighbors 
        
    return list(set(neighbors))
def get_PF(myArray):
    myArray = myArray[myArray[:,0].argsort()]
    # Add first row to pareto_frontier
    pareto_frontier = myArray[0:1,:]
    # Test next row against the last row in pareto_frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x]
                for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier

def get_ranks(graph_name, comm_name, measure):
    G = read_graph(filename=graph_name)
    if measure == 'page_rank':
        T = nx.pagerank(G, alpha = 0.85)
    elif measure == 'degree_centrality':
        T = degree_centrality(G)
    elif measure == 'katz_centrality':
        T = katz_centrality_numpy(G)
    elif measure == 'betweenness':
        T = betweenness_centrality(G)
    elif measure == 'eigenvector_centrality':
        T = eigenvector_centrality_numpy(G)
    elif measure == 'closeness':
        T = closeness_centrality(G)
    elif measure == 'core':
        G.remove_edges_from(nx.selfloop_edges(G))
        T = core_number(G)
    elif measure == 'two-hop':
        T = {}
        for node in G:
            T[node] = len(n_neighbor(G,node,2))


    data = pd.DataFrame()
    node = []
    centr = []

    for key,value in T.items():
        node.append(key)
        centr.append(value)

    rank = [x for x in range(1,len(node)+1)]
    data["node"] = node
    data["page_rank"] = centr

    data = data.sort_values(by='page_rank', ascending=False)
    data["overall_rank"] = rank

    community = pd.read_csv(comm_name,sep=",")

    int_df = pd.merge(data, community, how ='inner', on =['node'])
    n_comm = len(set(int_df["comm"].to_list()))
    node = []
    rank_comm = []
    len_list = []
    for i in range(n_comm):
        list_comm = int_df[int_df["comm"] == i+1]
        len_list.append(len(list_comm))
        for i in range(len(list_comm)):
            node.append(list_comm["node"].iloc[i])
            rank_comm.append(i+1)

    data_comm = pd.DataFrame()
    data_comm["rank_comm"] = rank_comm
    data_comm["node"] = node


    int_df = pd.merge(int_df, data_comm, how ='inner', on =['node'])
    
    return int_df



def upscaling_process(nodes, scaled_table, original_table, scale_original):
    solution = []
    NODES = nodes
    for item in nodes:
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split()  

        N = []
        for node in nodes_split:
            node = int(node)
            t = scaled_table.loc[scaled_table["node"] == node]
            if len(t) > 0:
                r = original_table[original_table["comm"] == int(t.comm)]
                n = r["node"].to_list()
                l = r["page_rank"].to_list()
                s = 0
                ii = 0
                while ii < int(scale_factor):
                    myArray = np.array(l)                
                    pos = (np.abs(myArray-float(t.rank_comm))).argmin()
                    if n[pos] in N:
                        l = np.delete(myArray, pos)
                        n = np.delete(n, pos)
                    else:
                        N.append(n[pos])
                        ii +=1
                        s +=1
                        l = np.delete(myArray, pos)
                        n = np.delete(n, pos)
                    
                    if len(l) == 0:
                        break
        # check if the new seed sets S has cardinality |s| * scaling_factor
        # This could happen if the real scaling_factor is not an int, but a float (e.g 4.2)
        # in this case, we cannot map every node to 4.2 nodes so we calculated it for 4 nodes
        # and then we add the rest of the nodes based on the overall rank
        if len(N) != int(len(nodes_split) * (scale_original)):
            k = int(len(nodes_split) * (scale_original)) - len(N)
            df_split = pd.DataFrame()
            df_split["node"] = [int(x) for x in nodes_split]
            df_new = pd.merge(scaled_table, df_split, on='node')

            n = df_new["node"].to_list()[:k]
            for node in n:
                node = int(node)
                t = scaled_table.loc[scaled_table["node"] == node]
                if len(t) > 0:
                    r = original_table[original_table["comm"] == int(t.comm)]
                    n = r["node"].to_list()
                    l = r["page_rank"].to_list()
                    s = 0
                    ii = 0
                    while ii < 1:
                        myArray = np.array(l)                
                        pos = (np.abs(myArray-float(t.rank_comm))).argmin()
                        if n[pos] in N:
                            l = np.delete(myArray, pos)
                            n = np.delete(n, pos)
                        else:
                            N.append(n[pos])
                            ii +=1
                            s +=1
                            l = np.delete(myArray, pos)
                            n = np.delete(n, pos)
                        
                        if len(l) == 0:
                            break
        solution.append(N)
    return solution


def get_propagation_results(G, args,idx1, idx2):
    original_filename = "graphs/{0}.txt".format(args["graph"])
    p = args["p"]
    no_simulations = args["no_simulations"]
    model = args["model"]
    G = read_graph(original_filename)

    nodes_ = []
    influence = []
    n_nodes = []
    logging.info('Propagation Process ....')

    for idx, item in enumerate(mapped_solution):	
        A = set(item)
        logging.debug('{0} / {1}'.format(idx+1,len(mapped_solution)))
        spread  = MonteCarlo_simulation(G, A, p, no_simulations, model, random_generator=None)
        influence.append(((spread[0] / G.number_of_nodes())* 100))
        nodes_.append(((len(A) / G.number_of_nodes())* 100))
        n_nodes.append(list(A))
    df_mapping = pd.DataFrame()
    df_mapping["n_nodes"] = nodes_
    df_mapping["influence"] = influence
    df_mapping["nodes"] = n_nodes
    df_mapping.to_csv('{0}_{1}_{2}-{3}-{4}-{5}.csv'.format(args["graph"],args["model"],args["s"],args["measure"], idx1, idx2), index=False)
    return df_mapping
def get_results(df,args):
    x =  df["n_nodes"].to_list()
    z = df["influence"].to_list()
    A_hv= []
    A_gd = []
    for i in range(len(x)):
        A_hv.append([-z[i],- (args["k"] - x[i])])
        A_gd.append([z[i], x[i]])
    
    return np.array(A_hv), np.array(A_gd)
def performance_indicators(df_mapping,filename_original_results, args,idx1, idx2):
    """
    Documentation: 
    
    """
    A_hv_map , A_gd_map = get_results(df_mapping, args)

    
    df_original = pd.read_csv(filename_original_results, sep=",")

    A_hv_original , A_gd_original = get_results(df_original, args)

    gd = get_performance_indicator("gd", A_gd_original)
    gd_distance = gd.do(A_gd_map)


    tot = 100 * args["k"] 
    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)
    hv_original = metric.do(A_hv_original) /tot
    hv_map = metric.do(A_hv_map) / tot

    logging.info('HyperArea (Hv/Hr): {0}'.format(hv_map/hv_original))
    logging.info('Generational Distance: {0}'.format(gd_distance))

    df_final = pd.DataFrame()
    df_final["measure"] = [args["measure"]]
    df_final["hyperarea"] = [hv_map/hv_original]
    df_final["generational_distance"] = [gd_distance]
    df_final.to_csv('experiments_upscaling/{0}_{1}_{2}_indicators-{3}-{4}.csv'.format(args["graph"],args["model"],args["s"],idx1, idx2), index=False)

#-------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    args = read_arguments() 

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  
    filename = "graphs_downscaled/{0}_{1}.txt".format(args["graph"], args["s"])
    scale_comm = "comm_ground_truth/{0}_{1}.csv".format(args["graph"], args["s"])
    filename_original = "graphs/{0}.txt".format(args["graph"])
    filename_original_comm = "comm_ground_truth/{0}.csv".format(args["graph"])
    G = read_graph(filename)
    G1 = read_graph(filename_original)

    scale_factor = int(G1.number_of_nodes() / G.number_of_nodes())
    scale_original = G1.number_of_nodes() / G.number_of_nodes()
    for idx1 in range(10):
        for idx2 in range(10):

            logging.info('Original Scaling Factor {0}'.format(args["s"])) 
            logging.info('Actual Scaling Factor {0}'.format(scale_original)) 

            
            df_scale_results = pd.read_csv("experiments_moea/{0}_{1}-{2}/run-{3}.csv".format(args["graph"],args["s"], args["model"], idx1+1),sep=",")
            df_scale_results = df_scale_results.sort_values(by="n_nodes", ascending=False)
            
            nodes = df_scale_results["nodes"].to_list()
            filename_original_results = "experiments_moea/{0}-{1}/run-{2}.csv".format(args["graph"], args["model"], idx2+1)

            try:
                scaled_table = pd.read_csv('experiments_upscaling/measure_groundtruth/{0}-{1}-{2}.csv'.format(args["graph"], args["s"], args["measure"]))
            except:
                logging.info('Calculating {} .....'.format(args["measure"]))
                scaled_table = get_ranks(filename, scale_comm, args["measure"])
            try:
                original_table = pd.read_csv('experiments_upscaling/measure_groundtruth/{0}-1-{1}.csv'.format(args["graph"], args["measure"]))
            except:
                logging.info('Calculating {} .....'.format(args["measure"]))
                original_table = get_ranks(filename_original, filename_original_comm, args["measure"])


            logging.info('{0} Done'.format(args["measure"]))
            
            mapped_solution = upscaling_process(nodes, scaled_table, original_table, scale_original)
            df_mapping = get_propagation_results(G, args, idx1+1, idx2+1)
            performance_indicators(df_mapping,filename_original_results, args,idx1+1, idx2+1)

            