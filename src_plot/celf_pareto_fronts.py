import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume

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

def get_hypervolume(df):
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x0 = []
    y0 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        x0.append(influence[idx])
        y0.append(n_nodes[idx])

    t = np.array([[x0[i],y0[i]] for i in range(len(x0))])
    t = get_PF(t)

    return t

def find_best(directory): 
    best_hv = 0
    best_file = ''
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')
            if best_hv < df['hv'].max(): 
                best_hv = df['hv'].max()
                best_file = file
    
    best_file = best_file.replace('_hv_', '')

    return best_hv, best_file


#---------------------------------------------------------------#

if __name__ == '__main__':
    directory = "exp1_out_facebook_combined_4-IC"
    
    hv_influence_seed, best_file = find_best(os.path.join(directory, "influence_seedSize"))
    df = pd.read_csv(os.path.join(directory, "influence_seedSize", best_file), sep = ',')
    pf_influence_seed = get_hypervolume(df)    
    
    hv_influence_seedSize_communities, best_file = find_best(os.path.join(directory, "influence_seedSize_time"))
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_time", best_file), sep = ',')
    pf_influence_seedSize_communities = get_hypervolume(df)    

    hv_influence_seedSize_communities_time, best_file = find_best(os.path.join(directory, "influence_seedSize_communities"))
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_communities", best_file), sep = ',')
    pf_influence_seedSize_communities_time = get_hypervolume(df)

    hv_influence_seedSize_time, best_file = find_best(os.path.join(directory, "influence_seedSize_communities_time"))
    df = pd.read_csv(os.path.join(directory, "influence_seedSize_communities_time", best_file), sep = ',')
    pf_influence_seedSize_time = get_hypervolume(df)

    print("hv_influence_seed:", hv_influence_seed)
    print("hv_influence_seedSize_communities:", hv_influence_seedSize_communities)
    print("hv_influence_seedSize_communities_time:", hv_influence_seedSize_communities_time)
    print("hv_influence_seedSize_time:", hv_influence_seedSize_time)

    plt.scatter(pf_influence_seed[:,0],pf_influence_seed[:,1], color='olive', label='influence_seed', facecolor='none', s=50)
    plt.scatter(pf_influence_seedSize_communities[:,0],pf_influence_seedSize_communities[:,1], color='brown', label='influence_seedSize_communities', marker='*',s=100)
    plt.scatter(pf_influence_seedSize_communities_time[:,0],pf_influence_seedSize_communities_time[:,1], color='black', label='influence_seedSize_communities_time', marker='.',s=100)
    plt.scatter(pf_influence_seedSize_time[:,0],pf_influence_seedSize_time[:,1], color='blue', label='influence_seedSize_time', marker='.',s=100)
    # plt.title('Comparing fitness functions', x=0.2, y=0.5,fontsize=12,weight="bold")
    plt.xlabel('% Influenced Nodes',fontsize=12)
    plt.ylabel('% Nodes as seed set',fontsize=12)
    plt.legend()

    plt.show()
    
    # graphs = ['soc-gemsec','soc-brightkite']

    # fig,(ax1, ax2) = plt.subplots(1,2, figsize=(10,3), sharey=True, sharex=False)
    # for index, name in enumerate(graphs):
    #     model = 'WC'
        
    #     df = pd.read_csv(f'experiments_upscaling/pfs_upscaling/{name}_WC_16-page_rank.csv', sep = ',')
    #     hv_16, pf16 = get_hypervolume(df)

    #     df = pd.read_csv(f'experiments_upscaling/pfs_upscaling/{name}_WC_32-page_rank.csv', sep = ',')
    #     hv_32, pf32 = get_hypervolume(df)

    #     df = pd.read_csv(f'experiments_heuristic/{name}_WC_CELF_runtime.csv', sep = ',')
    #     hv_celf, pf_celf = get_hypervolume(df)

    #     #print('CELF s=16', hv_16/hv_celf)
    #     #print('CELF s=32', hv_32/hv_celf)
        
    #     if index == 0:
    #         ax1.scatter(pf_celf[:,0],pf_celf[:,1] , color='olive', label='CELF', facecolor='none', s=50)
    #         ax1.scatter(pf16[:,0],pf16[:,1],color='brown', label='Upscaled Solutions $\it{s}$=16', marker='*',s=100)
    #         ax1.scatter(pf32[:,0],pf32[:,1],color='brown', label='Upscaled Solutions $\it{s}$=32', marker='.',s=100)
    #         ax1.set_xlim(0,29)
    #         ax1.set_title(f'{name}', x=0.2, y=0.5,fontsize=12,weight="bold")
    #         ax1.set_xlabel('% Influenced Nodes',fontsize=12)
    #         ax1.set_ylabel('% Nodes as seed set',fontsize=12)
    #     elif index==1:
    #         ax2.scatter(pf_celf[:,0],pf_celf[:,1] , color='olive', label='CELF', facecolor='none', s=50)
    #         ax2.scatter(pf16[:,0],pf16[:,1],color='brown', label='Upscaled Solutions $\it{s}$=16', marker='*',s=100)
    #         ax2.scatter(pf32[:,0],pf32[:,1],color='brown', label='Upscaled Solutions $\it{s}$=32', marker='.',s=100)
    #         ax2.set_xlim(0,43)
    #         ax2.legend(fontsize=10)
    #         ax2.set_title(f'{name}', x=0.2, y=0.5,fontsize=12,weight="bold")
    #         ax2.set_xlabel('% Influenced Nodes',fontsize=12)

    # plt.subplots_adjust(left=0.07,
    # bottom=0.15, 
    # right=0.99, 
    # top=0.98, 
    # wspace=0., 
    # hspace=0.0)

    # plt.savefig('Figure-8.eps', format='eps')
    # plt.show()
