import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import networkx as nx
from functools import partial

# local libraries
from src.load import read_graph
from src.utils import inverse_ncr
from moea import moea_influence_maximization
# spread models
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
# smart initialization
from src.smart_initialization import degree_random
from src.nodes_filtering.select_best_spread_nodes import filter_best_nodes as filter_best_spread_nodes
from src.nodes_filtering.select_min_degree_nodes import filter_best_nodes as filter_min_degree_nodes


def read_arguments():
    """
	Algorithm arguments, it is sufficient to specify all the parameters in the
	.json config file, which should be given as a parameter to the script, it
	should contain all the other script parameters.
	"""
    parser = argparse.ArgumentParser(
        description='Evolutionary algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--k', type=float, default=0.025, help='Seed set size as percentage of the whole network.')
    parser.add_argument('--p', type=float, default=0.01,
                        help='Probability of influence spread in the IC model.')
    parser.add_argument('--no_simulations', type=int, default=1,
                        help='Number of simulations for spread calculation'
                             ' when the Monte Carlo fitness function is used.')
    parser.add_argument('--model', default="WC", choices=['IC', 'WC'],
                        help='Influence propagation model.')

    parser.add_argument('--no_obj', default=2, type=int, choices= [2,3],
                        help='Number of objective functions')
    # EA setup.
    parser.add_argument('--population_size', type=int, default=100,
                        help='EA population size.')
    parser.add_argument('--offspring_size', type=int, default=100,
                        help='EA offspring size.')
    parser.add_argument('--max_generations', type=int, default=1000,
                        help='Generational budget.')
    parser.add_argument('--crossover_rate', type=float, default=1.0,
                        help='EA crossover rate.')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                        help='EA mutation rate.')
    parser.add_argument('--tournament_size', type=int, default=5,
                        help='EA tournament size.')
    parser.add_argument('--num_elites', type=int, default=2,
                        help='EA number of elite individuals.')
    parser.add_argument('--no_runs', type=int, default=10,
                        help='EA number of runs')
    parser.add_argument('--random_seed', type=int, default=10,
                        help='Seed to initialize the pseudo-random number '
                             'generation.')
    # EA improvements setup.
    # Smart initialization.
    parser.add_argument('--smart_initialization', default="degree_random",
                        choices=["none", "degree", "eigenvector", "katz",
                                 "closeness", "betweenness",
                                 "community", "community_degree",
                                 "community_degree_spectral", "degree_random",
                                 "degree_random_ranked"],
                        help='If set, an individual containing the best nodes '
                             'according to the selected centrality metric will '
                             'be inserted into the initial population.')
    parser.add_argument('--smart_initialization_percentage', type=float,
                        default=0.5,
                        help='Percentage of "smart" initial population, to be '
                             'specified when multiple individuals technique is '
                             'used.')
    parser.add_argument("--filter_best_spread_nodes", type=str, nargs="?",
                        const=True, default=True,
                        help="If true, best spread filter is used.")
    parser.add_argument("--search_space_size_min",
                        type=float,
                        #default=None,
                        default=1e9,
                        help="Lower bound on the number of combinations.")
    parser.add_argument("--search_space_size_max",
                        type=float,
                        default=1e11,
                        #default=None,
                        help="Upper bound on the number of combinations.")
    # Smart mutations.
    parser.add_argument('--mutation_operator',
                        type=str,
                        default="ea_global_random_mutation",
                        choices=[
                            "ea_global_random_mutation",
                            "ea_local_neighbors_random_mutation",
                            "ea_local_neighbors_second_degree_mutation",
                            "ea_global_low_spread",
                            "ea_global_low_deg_mutation",
                            "ea_local_approx_spread_mutation",
                            "ea_local_embeddings_mutation",
                            "ea_global_subpopulation_mutation",
                            "ea_adaptive_mutators_alteration",
                            "ea_local_neighbors_spread_mutation",
                            "ea_local_additional_spread_mutation",
                            "ea_local_neighbors_second_degree_mutation_emb",
                            "ea_global_low_additional_spread",
                            "ea_differential_evolution_mutation",
                            "ea_global_activation_mutation",
                            "ea_local_activation_mutation",
                            "adaptive_mutations",
                        ],
                        help="Mutation operator in case a single mutation is used.",
                        )
    parser.add_argument('--mutators_to_alterate',
                        type=str,
                        nargs='+',
                        default=[
                            "ea_one_point_crossover",
                        ],
                        help='List of mutation methods to alter, in case if '
                             'adaptive mutations are used.')
    # Graph setup.
    parser.add_argument('--graph', default='facebook_combined',
                        choices=['facebook_combined', 'fb_politician',
                                 'deezerEU', 'fb_org', 'fb-pages-public-figuree',
                                 'pgp', 'soc-gemsec', 'soc-brightkite'],
                        help='Graph name')
    parser.add_argument('--g_seed', type=int, default=None,
                        help='Random seed of the graph, in case if generated.')
    parser.add_argument('--g_file', default=None,
                        help='Location of the graph file.')
    parser.add_argument('--downscaled', type=bool, default=True,
                        help='Original or Downscaled Graph')
    parser.add_argument('--s', type=int, default=8,
                        help='Scaling factor if downscaled graph is used')  
    # Input/output setup.
    parser.add_argument('--config_file',
                        type=str,
                        help="Input json file containing the experimental setup: "
                             "arguments for the script.")
    parser.add_argument('--out_dir', 
                        #default='experiments/', 
                        default=None,
                        type=str,
                        help='Location of the output directory in case if '
                             'outfile is preferred to have a default name.')
    args = parser.parse_args()
    args = vars(args)
    if args["config_file"] is not None:
        with open(args["config_file"], "r") as f:
            in_params = json.load(f)
        ea_args = in_params["script_args"]
        ea_args["config_file"] = args["config_file"]
        if ea_args["out_dir"] is None:
            # Overwrite the parameter.
            ea_args["out_dir"] = args["out_dir"]
        # check whether all the parameters are specified in the config file
        if set(args.keys()) != set(ea_args.keys()):
            if len(set(args.keys()).difference(set(ea_args.keys()))) > 0:
                print("Missing arguments: {}".format(set(args.keys()).difference(set(ea_args.keys()))))
            else:
                print("Unknown arguments: {}".format(set(ea_args.keys()).difference(set(args.keys()))))
            raise KeyError("Arguments error")
        args.update(ea_args)
    return args

#------------------------------------------------------------------------------------------------------------#  

#SMART INITIALIZATION
def create_initial_population(G, args, prng=None, nodes=None):
    """
	Smart initialization techniques.
	"""
    # smart initialization
    initial_population = None

    initial_population = degree_random(args["k"], G,
                                        int(args["population_size"] * args["smart_initialization_percentage"]),
                                        prng, nodes=nodes)

    return initial_population
def filter_nodes(G, args):
    """
	Selects the most promising nodes from the graph	according to specified
	input arguments techniques.

	:param G:
	:param args:
	:return:
	"""
    # nodes filtering
    nodes = None
    if args["filter_best_spread_nodes"]:
        best_nodes = inverse_ncr(args["search_space_size_min"], args["k"])
        error = (inverse_ncr(args["search_space_size_max"], args["k"]) - best_nodes) / best_nodes
        filter_function = partial(MonteCarlo_simulation_max_hop, G=G, random_generator=prng, p=args["p"], model=args["model"],
                                  max_hop=3, no_simulations=1)
        nodes = filter_best_spread_nodes(G, best_nodes, error, filter_function)
    nodes = filter_min_degree_nodes(G, args["min_degree"], nodes)

    return nodes

#------------------------------------------------------------------------------------------------------------#

#LOCAL FUNCTIONS
def get_graph(args):
    if args["downscaled"]:
        filename = "scale_graphs/{0}_{1}.txt".format(args["graph"], args["s"])
    else:
        filename = 'graphs/{0}.txt'.format(args["graph"])
    graph_name = str(os.path.basename(filename))
    graph_name = graph_name.replace(".txt", "")

    G = read_graph(filename)

    return G, graph_name

def create_folder(args, graph_name):
    if args["out_dir"] != None:
            path = '{0}{1}-{2}'.format(args["out_dir"],graph_name,args["model"]) 
    else:
        path = '{0}-{1}'.format(graph_name,args["model"])
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    return path

def get_filter_nodes(args, G):
    my_degree_function = G.degree
    mean = []
    for item in G:
        mean.append(my_degree_function[item])
        
    #define minimum degree threshold as the average degree +1 
    args["min_degree"] = int(np.mean(mean)) + 1
    nodes_filtered = filter_nodes(G, args)
    return nodes_filtered

#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    args = read_arguments()
    G, graph_name = get_graph(args)
    prng = random.Random(args["random_seed"])
    
    #Calculate k based on network size
    args["k"] = int(G.number_of_nodes() * args["k"])
    
    #create directory for saving results
    path = create_folder(args, graph_name)
    
    #select best nodes with smart initiliazation
    nodes_filtered = get_filter_nodes(args,G)
    
    for run in range(args["no_runs"]):
        initial_population = create_initial_population(G, args, prng, nodes_filtered)


        #Print Graph's information and properties
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.info(nx.classes.function.info(G))
        
        
        file_path = 'run-{0}'.format(run+1)
        file_path = path+'/'+file_path                
        
        ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
        seed_sets = moea_influence_maximization(G, args, fitness_function=MonteCarlo_simulation, population_file=file_path,initial_population=initial_population)
        
