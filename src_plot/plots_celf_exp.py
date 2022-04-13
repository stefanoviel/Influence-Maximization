from telnetlib import TN3270E
import pandas as pd
import numpy as np
from pymoo.factory import get_performance_indicator
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
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




graphs = ['soc-gemsec','soc-brightkite']
fig,(ax1, ax2) = plt.subplots(1,2, figsize=(10,3), sharey=True, sharex=False)

for index, name in enumerate(graphs):
    model = 'WC'
    df = pd.read_csv(f'{name}_WC_16-page_rank.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x0 = []
    y0 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x0.append(influence[idx])
        y0.append(n_nodes[idx])





    t0 = np.array([[x0[i],y0[i]] for i in range(len(x0))])

    t0 = get_PF(t0)


    A = []
    for i in range(len(t0)):
        A.append(list([-t0[i][0],- (2.5 - t0[i][1])]))



    A = np.array(A)

    tot = 100 * 2.5 
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_MAP_16 = metric.do(A) / tot

    print(hv_MAP_16)

    df = pd.read_csv(f'{name}_WC_32-page_rank.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x = []
    y = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x.append(influence[idx])
        y.append(n_nodes[idx])





    t1 = np.array([[x[i],y[i]] for i in range(len(x))])

    t1 = get_PF(t1)


    A = []
    for i in range(len(t1)):
        A.append(list([-t1[i][0],- (2.5 - t1[i][1])]))



    A = np.array(A)

    tot = 100 * 2.5 
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_MAP = metric.do(A) / tot

    print(hv_MAP)




    df = pd.read_csv(f'{name}_high_degree_nodes_runtime_WC.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x1 = []
    y1 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x1.append(influence[idx])
        y1.append(n_nodes[idx])



    x_heu =  df["n_nodes"].to_list()
    z_heu = df["influence"].to_list()

    t2 = np.array([[x1[i],y1[i]] for i in range(len(x1))])

    t2 = get_PF(t2)



    A = []
    for i in range(len(t2)):
        A.append([-t2[i][0],- (2.5 - t2[i][1])])



    A = np.array(A)

    tot = 100 * 2.5
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_SD = metric.do(A) / tot

    print(hv_SD)

    print('High_degree_nodes s=32', hv_MAP_16/hv_SD)
    print('High_degree_nodes s=32', hv_MAP/hv_SD)

    df = pd.read_csv(f'{name}_low_distance_nodes_runtime_WC.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x2 = []
    y2 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x2.append(influence[idx])
        y2.append(n_nodes[idx])



    x_heu =  df["n_nodes"].to_list()
    z_heu = df["influence"].to_list()

    t3 = np.array([[x2[i],y2[i]] for i in range(len(x2))])

    t3 = get_PF(t3)

    A = []
    for i in range(len(t3)):
        A.append([-t3[i][0],- (2.5 - t3[i][1])])



    A = np.array(A)

    tot = 100 * 2.5
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_HD = metric.do(A) / tot

    print(hv_HD)
    print('Distance s=16', hv_MAP_16/hv_HD)
    print('Distance s=32', hv_MAP/hv_HD)

    df = pd.read_csv(f'{name}_single_discount_high_degree_nodes_runtime_WC.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x3 = []
    y3 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x3.append(influence[idx])
        y3.append(n_nodes[idx])



    x_heu =  df["n_nodes"].to_list()
    z_heu = df["influence"].to_list()


    A = []
    t4 = np.array([[x3[i],y3[i]] for i in range(len(x3))])

    t4 = get_PF(t4)
    for i in range(len(t4)):
        A.append([-t4[i][0],- (2.5 - t4[i][1])])



    A = np.array(A)

    tot = 100 * 2.5
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_SDD = metric.do(A) / tot

    print(hv_SDD)

    print('Single_discount s=16', hv_MAP_16/hv_SDD)
    print('Single_discount s=32', hv_MAP/hv_SDD)




    df = pd.read_csv(f'{name}_WC_CELF_runtime.csv', sep = ',')
    nodes = df["nodes"].to_list()
    influence = df['influence'].to_list()
    n_nodes = df["n_nodes"].to_list()
    x4 = []
    y4 = []
    for idx, item in enumerate(nodes):
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace(",","")
        nodes_split = item.split() 
        #print(, influence[idx])
        x4.append(influence[idx])
        y4.append(n_nodes[idx])



    x_heu =  df["n_nodes"].to_list()
    z_heu = df["influence"].to_list()


    A = []
    t5 = np.array([[x4[i],y4[i]] for i in range(len(x4))])

    t5 = get_PF(t5)
    for i in range(len(t5)):
        A.append([-t5[i][0],- (2.5 - t5[i][1])])



    A = np.array(A)

    tot = 100 * 2.5
    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)


    hv_CELF = metric.do(A) / tot

    print(hv_CELF)

    print('CELF s=16', hv_MAP_16/hv_CELF)
    print('CELF s=32', hv_MAP/hv_CELF)



    # plt.scatter(x1,y1, color='yellow', label='Single Discount Heuristic', facecolor='none')
    # plt.scatter(x2,y2, color='pink', label='CELF Heuristic')
    # plt.scatter(x2,y2, color='grey', label='Highest Degree Heuristic', facecolor='none')

    # plt.scatter(x,y, color='black', label='Mapping')


    #plt.scatter(t2[:,0],t2[:,1], color='purple', label='Highest Degree Heuristic', facecolor='none')
    #plt.scatter(t4[:,0],t4[:,1], color='olive', label='Single Discount Heuristic',facecolor='none')
    #plt.scatter(t3[:,0],t3[:,1] , color='grey', label='Low distance', facecolor='none')
    
    if index == 0:
        ax1.scatter(t5[:,0],t5[:,1] , color='olive', label='CELF', facecolor='none', s=50)
        ax1.scatter(t0[:,0],t0[:,1],color='brown', label='Upscaled Solutions $\it{s}$=16', marker='*',s=100)
        ax1.scatter(t1[:,0],t1[:,1],color='brown', label='Upscaled Solutions $\it{s}$=32', marker='.',s=100)

        ax1.set_xlim(0,29)
        #ax1.legend()
        ax1.set_title(f'{name}', x=0.2, y=0.5,fontsize=12,weight="bold")

        ax1.set_xlabel('% Influenced Nodes',fontsize=12)
        ax1.set_ylabel('% Nodes as seed set',fontsize=12)
    elif index==1:
        ax2.scatter(t5[:,0],t5[:,1] , color='olive', label='CELF', facecolor='none', s=50)
        ax2.scatter(t0[:,0],t0[:,1],color='brown', label='Upscaled Solutions $\it{s}$=16', marker='*',s=100)
        ax2.scatter(t1[:,0],t1[:,1],color='brown', label='Upscaled Solutions $\it{s}$=32', marker='.',s=100)
        ax2.set_xlim(0,43)
        ax2.legend(fontsize=10)
        ax2.set_title(f'{name}', x=0.2, y=0.5,fontsize=12,weight="bold")

        ax2.set_xlabel('% Influenced Nodes',fontsize=12)
        #ax2.set_ylabel('% Nodes as seed set',fontsize=12)    

plt.subplots_adjust(left=0.07,
bottom=0.15, 
right=0.99, 
top=0.98, 
wspace=0., 
hspace=0.0)

plt.savefig('prova.png', format='png')
plt.savefig('exp_vs_CELF.eps', format='eps')
plt.show()
exit(0)