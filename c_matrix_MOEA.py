import imp
from cv2 import rotate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r = ['Hyperarea']
model = ['IC']
scale = [8,4,2]
for indicator in r:
    graphs = ['deezerEU', 'pgp', 'fb_politician','fb-pages-public-figure', 'facebook_combined']
    MAP = {}
    for item in graphs:
        MAP[item] = []


    print(MAP)
    for item in graphs:
        for m in model:
            file = item + '-' + m
            df = pd.read_csv('matrix_MOEA_results/'+file)
            print(df)
            x = df['HyperArea']
            for value in x:
                MAP[item].append(value)
    print(MAP)

    conf_arr = []

    for key, value in MAP.items():
        conf_arr.append(MAP[key])


    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    my_cmap ='YlGn'  
    res = ax.imshow(np.array(conf_arr),cmap = my_cmap,vmin=0, vmax=1)
    height = len(conf_arr[0])
    width = len(conf_arr)

    for x in range(width):
        for y in range(height):
                ax.annotate(str(round(conf_arr[x][y],2)), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
    cb = fig.colorbar(res)

    final_names = []
    for item in model:
        for s in scale:
            final_names.append(item + '-' + str(s))
    plt.yticks(np.arange(len(graphs)), graphs)
    plt.xticks(np.arange(len(model) * len(scale)), final_names, rotation=70)
    plt.title('MOEA')
    #plt.savefig('confusion_matrix.png', format='png')
    plt.show()