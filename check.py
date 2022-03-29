from telnetlib import TN3270E
import pandas as pd
import numpy as np
from pymoo.factory import get_performance_indicator
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


df = pd.read_csv('prova-fb-pages-artist_IC_32-page_rank.csv', sep = ',')
nodes = df["nodes"].to_list()
print(type(nodes[0]))
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


x_mapping =  df["n_nodes"].to_list()
z_mapping = df["influence"].to_list()


A = []
for i in range(len(x_mapping)):
    A.append([-z_mapping[i],- (2.5 - x_mapping[i])])



A = np.array(A)

tot = 100 * 2.5 
from pymoo.indicators.hv import Hypervolume

metric = Hypervolume(ref_point= np.array([0,0]),
                    norm_ref_point=False,
                    zero_to_one=False)


hv_MAP = metric.do(A) / tot

print(hv_MAP)





df = pd.read_csv('heuristics_experiment/high_degree_nodes_IC.csv', sep = ',')
nodes = df["nodes"].to_list()
print(type(nodes[0]))
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


A = []
for i in range(len(x_heu)):
    A.append([-z_heu[i],- (2.5 - x_heu[i])])



A = np.array(A)

tot = 100 * 2.5
from pymoo.indicators.hv import Hypervolume

metric = Hypervolume(ref_point= np.array([0,0]),
                    norm_ref_point=False,
                    zero_to_one=False)


hv_HEU = metric.do(A) / tot

print(hv_HEU)

print('FINAL', hv_MAP/hv_HEU)
import matplotlib.pyplot as plt


#t1 = np.array([[x[i],y[i]] for i in range(len(x))])

#t1 = get_PF(t1)

#t2 = np.array([[x1[i],y1[i]] for i in range(len(x1))])

#t2 = get_PF(t1)


plt.scatter(x1,y1, color='blue', label='Single Discount Heuristic', facecolor='none')
plt.scatter(x,y, color='red', label='Mapping')

plt.xlim(0,100)
plt.legend()

plt.savefig('prova.png', format='png')