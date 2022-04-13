import sys
import powerlaw
import numpy as np
import pandas as pd

import warnings
import collections
import matplotlib.pyplot as plt

sys.path.insert(0, '')
from src.load import read_graph

warnings.filterwarnings("ignore")

def plot_degree_distribution(graphs,alias):
    fig,a =  plt.subplots(2,3, sharey=True, sharex=True, figsize=(9,6))
    ax = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]

    for idx, name in enumerate(graphs):
        filenames = ["graphs_downscaled/{0}_8.txt".format(name),"graphs_downscaled/{0}_4.txt".format(name),"graphs_downscaled/{0}_2.txt".format(name),"graphs/{0}.txt".format(name)]    

        x = ['Original', '$\it{s}$=2', '$\it{s}$=4', '$\it{s}$=8']
        filenames = filenames[::-1]
        color = ["green", 'blue','orange','red']
        color = color[::-1]
        i = 0
        t = ax[idx]
        
        list_leg = []
        for item in filenames:
            G = read_graph(item)
            degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
            degreeCount = collections.Counter(degree_sequence)
            x1, y = zip(*degreeCount.items())                                                                                                                      

            a[t[0]][t[1]].set_xscale('log')  
            a[t[0]][t[1]].set_yscale('log')                                                                                                        
            if idx == 5:    
                    k = a[t[0]][t[1]].scatter(x1, y, marker='x', s=30,color=color[i], label = str(x[i]))
                    list_leg.append(k)
                    a[t[0]][t[1]].legend(bbox_to_anchor=(0.5,0.9), fontsize=12)

            else:
                a[t[0]][t[1]].scatter(x1, y, marker='x', s=30,color=color[i])
            i = i+1
        
        a[t[0]][t[1]].set_title(alias[idx], x=0.5, y=0.9, fontsize=12)
        a[t[0]][t[1]].xaxis.get_label().set_fontsize(12)
        a[t[0]][t[1]].yaxis.get_label().set_fontsize(12)
    
    plt.setp(a[-1, :], xlabel='Degree')
    plt.setp(a[:, 0], ylabel='Frequency')
    plt.subplots_adjust(left=0.07,
                bottom=0.1, 
                right=0.99, 
                top=0.99, 
                wspace=0, 
                hspace=0)
    plt.savefig('Figure-2.eps', format='eps')
    plt.show()


def get_alpha_powerlaw(graphs,alias):
    power_law = {}
    for idx, name in enumerate(graphs):
        filenames = ["graphs_downscaled/{0}_8.txt".format(name),"graphs_downscaled/{0}_4.txt".format(name),"graphs_downscaled/{0}_2.txt".format(name),"graphs/{0}.txt".format(name)]    
        power_law[alias[idx]] = []
        
        for item in filenames:
            G = read_graph(item)
            degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
            fit = powerlaw.Fit(degree_sequence, xmin=min(degree_sequence), xmax=max(degree_sequence))
            power_law[alias[idx]].append(round(fit.power_law.alpha,2))

    df = pd.DataFrame.from_dict(power_law, orient='index')
    #uncomment if you need the latex table with the alpha values
    #df.to_latex('powerlaw_table',index=True)
    print(df)


if __name__ == '__main__':
    graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
    
    alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
    plot_degree_distribution(graphs, alias)
    get_alpha_powerlaw(graphs,alias)
