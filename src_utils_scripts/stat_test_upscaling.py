import sys
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

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
    parser.add_argument('--measure', default='page_rank',
                        choices=['two-hop','page_rank', 'degree_centrality',
                                'katz_centrality','betweenness', 'closeness', 
                                'eigenvector_centrality', 'core'])
    parser.add_argument('--test', type=str, default='mannwhitneyu',
                        help='Statistical Test', choices = ['wilcoxon','mannwhitneyu'])  

    parser.add_argument('--model', type=str, default='IC', choices = ['IC', 'WC'],
                        help='Propagation Model')  
    args = parser.parse_args()
    args = vars(args)

    return args


if __name__ == '__main__':
    args = read_arguments() 
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    hv_upscaling = []

    for idx1 in range(10):
        df = pd.read_csv('experiments_upscaling/{0}_{1}_{2}_MAPPING_{3}-{4}.csv'.format(args["graph"], args["model"], args["s"], idx1+1, 1), sep=',')
        pg = df[df["measure"] == args["measure"]].Hyperarea.item()
        hv_upscaling.append(pg)

    hv_moea = []
    for idx1 in range(10):
        filename_original_results = "experiments_moea/{0}-{1}/run-{2}_hv_.csv".format(args["graph"], args["model"],idx1+1)
        df = pd.read_csv(filename_original_results, sep= ',')
        hv = df[df['generation'] == max(df['generation'])].hv.item()
        hv_moea.append(hv)

    if args["test"] == 'wilcoxon':
        res = wilcoxon(hv_moea, hv_upscaling)
    elif args["test"] == 'mannwhitneyu':
        res = mannwhitneyu(hv_moea, hv_upscaling)

    
    logging.info('p-value: {0}'.format(res.pvalue))

