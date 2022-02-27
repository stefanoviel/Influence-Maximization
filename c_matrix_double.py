import imp
from cv2 import rotate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
_, axs = plt.subplots(1, 2, figsize=(8, 8))
axs = axs.flatten()
r = ['Hyperarea', 'GD']
model = ['IC','WC']
scale = [8,4,2]
for idx, indicator in enumerate(r):
    graphs = ['pgp', 'fb_politician','fb-pages-public-figure', 'facebook_combined', 'fb_org']
    MAP = {}
    for item in graphs:
        MAP[item] = []


    print(MAP)
    for item in graphs:
        for m in model:
            for s in scale:
                file = item + '_' + m
                df = pd.read_csv(file+ '_' + str(s) + '_MAPPING.csv')
                r = (df[df["measure"] == 'page_rank'])
                r = r[indicator].to_list()
                MAP[item].append(r[0])
    print(MAP)
    conf_arr = []

    for key, value in MAP.items():
        conf_arr.append(MAP[key])


    # fig = plt.figure(figsize=(12,6)) 
    # plt.clf()
    # ax = fig.add_subplot(111)
    # ax.set_aspect(1)

    my_cmap ='OrRd'  
    #res = ax.imshow(np.array(conf_arr),cmap = my_cmap,vmin=min(min(conf_arr)), vmax=max(max(conf_arr)))
    
    axs[idx].imshow(np.array(conf_arr),cmap = my_cmap,vmin=0, vmax=1)

    height = len(conf_arr[0])
    width = len(conf_arr)

    for x in range(width):
        for y in range(height):
                axs[idx].annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
    #cb = fig.colorbar(res)

    final_names = []
    for item in model:
        for s in scale:
            final_names.append(item + ' s=' + str(s))
    
    if idx != 0:
        axs[idx].set_yticks(np.arange(len(graphs)), graphs)

    axs[idx].set_xticks(np.arange(len(model) * len(scale)), labels= final_names)
    axs[idx].set_title('Mapping - Page Rank - ' + indicator)
    #plt.savefig('plot_mapping/matrixFINAL.jpeg', dpi=250)
    #plt.show()

plt.show()