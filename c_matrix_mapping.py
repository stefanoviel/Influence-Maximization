import imp
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

r = ['Hyperarea']
name_graph = ['fb_org', 'fb_politician', 'fb-pages-public-figure', 'pgp', 'facebook_combined']
for graph in name_graph:
    for indicator in r:
        degree_measure = ['two-hop','page_rank', 'degree_centrality','katz_centrality', 'betweenness', 'closeness', 'eigenvector_centrality', 'core']
        MAP = {}
        for item in degree_measure:
            MAP[item] = []
    
        filenames = [graph +'_WC_8_MAPPING.csv', graph +'_WC_4_MAPPING.csv', graph +'_WC_2_MAPPING.csv',graph +'_IC_8_MAPPING.csv',graph +'_IC_4_MAPPING.csv',graph +'_IC_2_MAPPING.csv']
        for item in filenames:
            df = pd.read_csv(item, sep=",")
            measure = df["measure"].to_list()
            hv = df[indicator].to_list()
            for idx,m in enumerate(measure):
                MAP[m].append(hv[idx])

        conf_arr = []

        for key, value in MAP.items():
            conf_arr.append(MAP[key])


        fig = plt.figure(figsize=(10,7)) 
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        if indicator == 'GD':
            #conf_arr = [[i / max(j) for i in j] for j in conf_arr]
            my_cmap ="YlOrBr"  
            my_cmap = plt.cm.get_cmap('Blues_r')
            res = ax.imshow(np.array(conf_arr),cmap = my_cmap,vmin=min(min(conf_arr)), vmax=(max(max(conf_arr))))
            best = smallesttInColumn(conf_arr, 8, 6)
            print(best)
            height = len(conf_arr[0])
            width = len(conf_arr)

            for x in range(width):
                for y in range(height):
                    if conf_arr[x][y] == best[y]:
                        ax.annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                                    horizontalalignment='center',
                                    verticalalignment='center', color='green')
                    else:
                        ax.annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                                    horizontalalignment='center',
                                    verticalalignment='center')
        else:
            my_cmap ="OrRd"  
            res = ax.imshow(np.array(conf_arr),cmap = my_cmap,vmin=0, vmax=1)
            best = largestInColumn(conf_arr, 8, 6)
            print(best)
            height = len(conf_arr[0])
            width = len(conf_arr)

            for x in range(width):
                for y in range(height):
                    if conf_arr[x][y] == best[y]:
                        ax.annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                                    horizontalalignment='center',
                                    verticalalignment='center', color='green')
                    else:
                        ax.annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                                    horizontalalignment='center',
                                    verticalalignment='center')
        cb = fig.colorbar(res)

        final_names = []
        for item in filenames:
            print(item)
            name = item.replace('_MAPPING.csv','')
            name = name.replace(graph,'')
            name = name.replace('_',' ')
            name = name.split()
            name = '{0} s={1}'.format(name[0], name[1])
            final_names.append(name)

        plt.yticks(np.arange(8), degree_measure)
        plt.xticks(np.arange(6), final_names, rotation=70)
        plt.title(graph)
        plt.savefig('plot_mapping/'+graph +'.jpeg', dpi=250)
        #plt.show()