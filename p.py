from cmath import log10
from fileinput import filename
from pickle import TRUE
import pandas as pd
import networkx as nx
from sklearn.cluster import MeanShift
from src.load import read_graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import collections
warnings.filterwarnings("ignore")
from math import log, log10
import math

LAW = {}
graphs = ['pgp', 'fb_politician','fb-pages-public-figure', 'facebook_combined', 'fb_org', 'deezerEU']
fig,a =  plt.subplots(2,3, sharey=True, sharex=True, figsize=(8,6))
ax = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]

for idx, name in enumerate(graphs):
    print(name)
    filenames = ["scale_graphs/{0}_8.txt".format(name),"scale_graphs/{0}_4.txt".format(name),"scale_graphs/{0}_2.txt".format(name),"graphs/{0}.txt".format(name)]    #filenames = ["scale_graphs/facebook_combined_prova8.txt","scale_graphs/facebook_combined_8.txt","graphs/facebook_combined.txt"]
    kk = []
    LAW[name] = []

    for item in filenames:
        G = read_graph(item)
        #print(nx.info(G))
        den = (2*G.number_of_edges()) / (G.number_of_nodes()*(G.number_of_nodes()-1))
        #print("Density --> {0}".format(den))
        my_degree_function = G.degree
        mean = []
        mean_degree = []
        for item in G:
            mean.append(my_degree_function[item])
        kk.append(mean)

#    for item in kk:
#        print(np.mean(item))


    x = ['Original', 's=2', 's=4', 's=8']
    #x = x[::-1]
    filenames = filenames[::-1]
    real = np.mean(kk[0])
    color = ["green", 'blue','orange','red']
    color = color[::-1]
    # plt.figure(figsize=(6, 6)) 
    # i = 0
    # for item in filenames:
    #     G = read_graph(item)   
    #     degree_freq = nx.degree_histogram(G)
    #     degrees = range(len(degree_freq))
    #     plt.loglog(degrees, degree_freq,'go-', color=color[i], label = str(x[i]))
    #     i +=1
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.savefig('degree_log.png')

    # #plt.show()
    # plt.cla()
    # plt.close()


    i = 0
    #fig, axs = plt.subplots(4, sharex=True,figsize=(8, 8)) 
    #fig.suptitle('Degree Distribution')
    '''
    i = 0
    for item in filenames:
        G = read_graph(item)
        degree_sequence = sorted([(d) for n, d in G.degree()], reverse=True)  # degree sequence
        degree_sequence_1 = sorted([log10(d) for n, d in G.degree()], reverse=True)  # degree sequence

        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        axs[len(filenames)-1-i].hist(degree_sequence, bins=int(max(degree_sequence)), facecolor=color[i], alpha=0.75, edgecolor='black', linewidth=0.5,log=True)

        i += 1

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.savefig('aaaa')
    plt.show()
    '''

    i = 0
    t = ax[idx]
    print(t)
    check = True
    
    list_leg = []
    for id, item in enumerate(filenames):
        G = read_graph(item)
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
        
        
        degreeCount = collections.Counter(degree_sequence)
        #print(degreeCount)
        x1, y = zip(*degreeCount.items())                                                                                                                      

        a[t[0]][t[1]].set_xscale('log')  
        a[t[0]][t[1]].set_yscale('log')                                                                                                        
        if idx == 5:    
                k = a[t[0]][t[1]].scatter(x1, y, marker='.',  s=25,color=color[i], label = str(x[i]))
                list_leg.append(k)
        else:
            a[t[0]][t[1]].scatter(x1, y, marker='.',  s=25,color=color[i])
        import powerlaw
        #y = [log10(x) for x in y]
        #x1 = [log10(x) for x in x1]
        import powerlaw
        fit = powerlaw.Fit(y, discrete=False)
        #fit.power_law.plot_pdf(color= 'b',linestyle='--',label='fit ccdf')

        print('alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)
        #a = fit.power_law.alpha
        #from scipy.stats import powerlaw as pw
        LAW[name].append(round(fit.power_law.alpha,2))
        #x = np.linspace(pw.ppf(0, a),pw.ppf(1, a), 1000)
        #a[idx].plot(x, pw.pdf(x, a), 'r-', lw=5, alpha=0.6, label='powerlaw pdf',color=color[i])
        #a[idx].show()
        i = i+1
    #a[t[0]][t[1]].set_xlabel('Degree')
    #a[t[0]][t[1]].set_ylabel('Frequency')  
    #a[t[0]][t[1]].legend()
    #plt.savefig(name + '_scatter')
    a[t[0]][t[1]].set_title(name, size=10)
    #plt.close()
    #plt.cla()



labels = [l.get_label() for l in list_leg]
print(labels)

plt.setp(a[-1, :], xlabel='Degree')
plt.setp(a[:, 0], ylabel='Frequency')
#plt.legend()

#plt.legend(list_leg, labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
#plt.tight_layout()

fig.legend(list_leg, labels,loc='upper center',ncol=4)
plt.subplots_adjust(left=0.08,
            bottom=0.1, 
            right=0.96, 
            top=0.85, 
            wspace=0.05, 
            hspace=0.2)
plt.show()
print(LAW)

df = pd.DataFrame.from_dict(LAW, orient='index')
df.to_latex('A',index=True)
print(df)
