import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_PF(df):
    myArray = df[df.columns[:2]].values
    myArray = myArray[myArray[:,0].argsort()]
    
    # Add first row to pareto_frontier
    pareto_frontier = myArray
    # print(pareto_frontier)
    # Test next row against the last row in pareto_frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x]
                for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier


def find_best(directory, hv_name): 
    """
    Find the run with the highest HV, we will consider the individual from its PF
    """
    best_hv = 0
    best_file = ''
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')
            if best_hv < df[ hv_name].max(): 
                best_hv = df[ hv_name].max()
                best_file = file
    
    best_file = best_file.replace('_hv_', '')

    return best_hv, best_file

def pfs_seedSize(directory): 
    hv_influence_seed, best_file = find_best(os.path.join(directory, "influence_seedSize"), 'hv_influence_seedSize')
    df = pd.read_csv(os.path.join(directory, "influence_seedSize", best_file), sep = ',')
    pf_influence_seed = get_PF(df)   
    lista = list(pf_influence_seed)
    lista = [list(l) for l in lista]
    lista.append([25.13, 0.1545])
    pf_influence_seed = np.array(lista)
    print(pf_influence_seed)
    
    hv_influence_seedSize_time , best_file = find_best(os.path.join(directory, "influence_seedSize_time"), "hv_influence_seedSize_time")
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_time", best_file), sep = ',')
    pf_influence_seedSize_time = get_PF(df)    

    hv_influence_seedSize_communities, best_file = find_best(os.path.join(directory, "influence_seedSize_communities"), "hv_influence_seedSize_communities")
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_communities", best_file), sep = ',')
    pf_influence_seedSize_communities = get_PF(df)

    hv_influence_seedSize_communities_time, best_file = find_best(os.path.join(directory, "influence_seedSize_communities_time"), "hv_influence_seedSize_communities_time")
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_communities_time", best_file), sep = ',')
    pf_influence_seedSize_communities_time = get_PF(df)

    print("hv_influence_seed:", hv_influence_seed)
    print("hv_influence_seedSize_communities:", hv_influence_seedSize_communities)
    print("hv_influence_seedSize_communities_time:", hv_influence_seedSize_communities_time)
    print("hv_influence_seedSize_time:", hv_influence_seedSize_time)

    fig = plt.scatter(pf_influence_seed[:,0], pf_influence_seed[:,1],   label='influence_seed',  s=50)
    # plt.scatter(pf_influence_seedSize_communities[:,0],pf_influence_seedSize_communities[:,1], color='brown', label='influence_seedSize_communities', marker='*',s=100)
    # plt.scatter(pf_influence_seedSize_communities_time[:,0],pf_influence_seedSize_communities_time[:,1], color='black', label='influence_seedSize_communities_time', marker='.',s=100)
    # plt.scatter(pf_influence_seedSize_time[:,0],pf_influence_seedSize_time[:,1], color='blue', label='influence_seedSize_time', marker='.',s=100)
    # plt.title('Comparing fitness functions', x=0.2, y=0.5,fontsize=12,weight="bold")
    # fig.set_sizes((12, 8))
    # plt.title(directory.replace('exp1_out_', ''))
    plt.ylabel('% Nodes as seed set',fontsize=12)
    plt.xlabel('% Influenced Nodes',fontsize=12)
    plt.legend()

    plt.show()

def pfs_no_seedSize(directory): 
    label = directory.replace('exp1_out_', '')
    print(label)
    hv_influence_seed, best_file = find_best(os.path.join(directory, "influence_seedSize"), 'hv_influence_seed')
    df = pd.read_csv(os.path.join(directory, "influence_seedSize", best_file), sep = ',')
    pf_influence_seed = get_PF(df)  
    print("hv_influence_seed:", hv_influence_seed)  
    plt.scatter(pf_influence_seed[:,0],pf_influence_seed[:,1], color='olive', label='influence_seed', facecolor='none', s=50)
    plt.title('influence_seedSize')
    plt.legend()
    plt.show()


    hv_influence_time, best_file = find_best(os.path.join(directory, "influence_time"), "hv_influence_time")
    df = pd.read_csv(os.path.join(directory, "influence_time", best_file), sep = ',')
    pf_influence_time = get_PF(df)    
    print("hv_influence_time:", hv_influence_time)
    plt.scatter(pf_influence_time[:,0],pf_influence_time[:,1], color='blue', label='influence_time', marker='.',s=100)
    plt.title('influence_time ' + label)
    plt.ylabel('1/time',fontsize=12)
    plt.xlabel('% Influenced Nodes',fontsize=12)
    plt.legend()
    plt.savefig('result_comparison/' + 'influence_time_' + label )
    plt.show()

    hv_influence_communities, best_file = find_best(os.path.join(directory, "influence_communities"), "hv_influence_communities")
    df = pd.read_csv(os.path.join(directory, "influence_communities", best_file), sep = ',')
    pf_influence_communities = get_PF(df)
    print("hv_influence_communities:", hv_influence_communities)
    plt.scatter(pf_influence_communities[:,0],pf_influence_communities[:,1], color='brown', label='influence_communities', marker='*',s=100)
    plt.title('influence_communities ' + label)
    plt.ylabel('# communities',fontsize=12)
    plt.xlabel('% Influenced Nodes',fontsize=12)
    plt.legend()
    plt.savefig('result_comparison/' + 'influence_communities_' + label )
    plt.show()


#---------------------------------------------------------------#

if __name__ == '__main__':
    for directory in os.listdir(): 

        # directory = "exp1_out_pgp_4-IC"
        if 'exp1_out_facebook_combined_4-IC-0.05' in directory: 
            # try: 
            print(directory)
            pfs_seedSize(directory)
            # except: 
                # print('something went wrong', directory)
        
    
    