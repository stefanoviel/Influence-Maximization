import imp
from tkinter import font
from cv2 import normalize, rotate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r = ['Hyperarea', 'GD']
model = ['IC','WC']
scale = [8,4,2]


fig,(ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(15,7))


ax = [[0,0], [0,1]]

for idk, indicator in enumerate(r):
    graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
    alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
    MAP = {}
    LATEX = {}
    for item in alias:
        MAP[item] = []
        LATEX[item] = []


    print(MAP)
    for idx, item in enumerate(graphs):
        for m in model:
            file = item + '-' + m
            df = pd.read_csv('matrix_MOEA_results/'+file)
            print(file)
            #print(df)
            x = df[indicator]
            x = x[::-1]
            for value in x:
                MAP[alias[idx]].append(value)
                LATEX[alias[idx]].append(round(value,2))
            
    
    #print(MAP)
    #MAP2= {}
    #for key, value in MAP.items():
    #    appo = key
    #    key = key.replace('_centrality' , '')
    #    key = key.replace('_' , ' ')
    #    MAP2[key] = value
        
    #print(len(ax), idk)
    t = ax[idk]
    df = pd.DataFrame.from_dict(MAP, orient='index')
    print(df, max(df))
    import seaborn as sns; sns.set_theme()
    labels = ['IC s=2','IC s=4','IC s=8','WC s=2','WC s=4','WC s=8']
    print(labels)
    if idk == 1:
        sns.heatmap(df, annot=True,cmap ="YlGnBu_r",vmin=0, vmax=30,linecolor='white', linewidths=.1, ax= ax2,cbar=True,annot_kws={"fontsize":20})    
        ax2.set_title('Generational Distance', fontsize=20)
        ax2.set_xticklabels(labels, rotation=90,fontsize=20)
        # column_max = df.idxmin(axis=0)
        # for col, variable in enumerate(df):
        #     position = df.index.get_loc(column_max[variable])
        #     ax2.add_patch(Rectangle((col, position),1,1, fill=False, edgecolor='red', lw=3))
    else:    
        sns.heatmap(df, annot=True,cmap ="YlGnBu",vmin=0, vmax=1,  linecolor='white', linewidths=.1,ax= ax1,cbar=True, annot_kws={"fontsize":20})    
        ax1.set_title('Hyperarea',fontsize=20)
        ax1.set_xticklabels(labels,rotation=90, fontsize=20)
        ax1.set_yticklabels(alias,rotation=0,fontsize=20)
        #ax1.set_ylabel('Dataset')
        from matplotlib.patches import Rectangle
        # column_max = df.idxmax(axis=0)
        # for col, variable in enumerate(df):
        #     position = df.index.get_loc(column_max[variable])
        #     ax1.add_patch(Rectangle((col, position),1,1, fill=False, edgecolor='white', lw=3))

    conf_arr = []

    for key, value in MAP.items():
        conf_arr.append(MAP[key])


    final_names = []
    for item in model:
        for s in scale:
            final_names.append(item + ' s=' + str(s))
    df = pd.DataFrame.from_dict(LATEX, orient='index', columns=final_names)
    df.to_latex('A',index=True)
    print(df)
    height = len(conf_arr[0])
    width = len(conf_arr)
    

    '''
    plt.yticks(np.arange(len(graphs)), graphs)
    plt.xticks(np.arange(len(model) * len(scale)), final_names, rotation=90)
    plt.title('MOEA SCALING ' + indicator)
    plt.savefig('plot_mapping/matrix_MOEA.jpeg', dpi=250)
    plt.show()
    '''
    #plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5],labels, rotation=90)
plt.subplots_adjust(left=0.08,
            bottom=0.16, 
            right=0.99, 
            top=0.95, 
            wspace=0.05, 
            hspace=0.05)

plt.savefig('MOEA_RESULTS.eps', format='eps')
plt.show()