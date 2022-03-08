import imp
from pprint import pprint
from cv2 import rotate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def largestInColumn(mat, rows, cols):
    best = []
    for i in range(cols):
         
        # initialize the maximum element with 0
        maxm = mat[0][i]
        # run the inner loop for news
        for j in range(rows):
             
            # check if any elements is greater
            # than the maximum elements
            # of the column and replace it
            if mat[j][i] > maxm:
                maxm = mat[j][i]
         
        # print the largest element
        # of the column
        best.append(maxm)
    return best

def smallesttInColumn(mat, rows, cols):
    best = []
    for i in range(cols):
         
        # initialize the maximum element with 0
        maxm = mat[0][i]
        # run the inner loop for news
        for j in range(rows):
             
            # check if any elements is greater
            # than the maximum elements
            # of the column and replace it
            if mat[j][i] < maxm:
                maxm = mat[j][i]
         
        # print the largest element
        # of the column
        best.append(maxm)
    return best

r = ['GD']
name_graph = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
fig,axn = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(12,7))
cbar_ax = fig.add_axes([.91, .3, .02, .4])
ax = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]

for idk, graph in enumerate(name_graph):

    degree_measure = ['degree_centrality','closeness', 'betweenness', 'eigenvector_centrality', 'katz_centrality','page_rank','core']
    MAP = {}
    MAP['MOEA'] = []
    for item in degree_measure:
        MAP[item] = []
    model = ['IC', 'WC']
    for m in model:
        file = graph + '-' + m
        df = pd.read_csv('matrix_MOEA_results/'+file)
        print(file)
        #print(df)
        x = df['Hyperarea']
        x = x[::-1]
        for value in x:
            MAP['MOEA'].append(value)
    filenames = [graph +'_IC_2_MAPPING.csv', graph +'_IC_4_MAPPING.csv', graph +'_IC_8_MAPPING.csv',graph +'_WC_2_MAPPING.csv',graph +'_WC_4_MAPPING.csv',graph +'_WC_8_MAPPING.csv']
    for item in filenames:
        df = pd.read_csv(item, sep=",")
        measure = df["measure"].to_list()
        hv = df['Hyperarea'].to_list()
        print(measure)
        for idx,m in enumerate(measure):
            if m in degree_measure:
                MAP[m].append(hv[idx])

    conf_arr = []
    print(MAP)
    for key, value in MAP.items():
        conf_arr.append(MAP[key])


    
    MAP2= {}
    for key, value in MAP.items():
        appo = key
        key = key.replace('_centrality' , '')
        key = key.replace('_' , ' ')
        MAP2[key] = value
        
    print(len(ax), idk)
    t = ax[idk]
    df = pd.DataFrame.from_dict(MAP2, orient='index')
    print(df)
    import seaborn as sns; sns.set_theme()
    if idk == 5:
        sns.heatmap(df, annot=True,cmap ="YlGnBu",vmin=0, vmax=1, linecolor='white', linewidths=.1, ax= axn[t[0]][t[1]],cbar=True, cbar_ax=cbar_ax,annot_kws={"fontsize":12})
    else:
        sns.heatmap(df, annot=True,cmap ="YlGnBu",vmin=0, vmax=1, linecolor='white', linewidths=.1, ax= axn[t[0]][t[1]],cbar=False, annot_kws={"fontsize":12})
    column_max = df.idxmax(axis=0)
    from matplotlib.patches import Rectangle
    for col, variable in enumerate(df):
        position = df.index.get_loc(column_max[variable])
        axn[t[0]][t[1]].add_patch(Rectangle((col, position),1,1, fill=False, edgecolor='red', lw=1))
    
    axn[t[0]][t[1]].set_title(alias[idk], fontsize=14)
    labels = ['IC s=2','IC s=4','IC s=8','WC s=2','WC s=4','WC s=8']
    axn[t[0]][t[1]].set_xticklabels(labels,rotation=90,fontsize=14)
    axn[t[0]][t[1]].set_yticklabels(list(MAP2.keys()),rotation=0,fontsize=14)
plt.subplots_adjust(left=0.07,
            bottom=0.1, 
            right=0.99, 
            top=0.95, 
            wspace=0.1, 
            hspace=0.1)
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('mapping_BEST.eps', format='eps')
plt.show()
    
