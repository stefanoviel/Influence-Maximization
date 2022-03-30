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



model = 'WC'
df = pd.read_csv(f'fb-pages-artist_{model}_32-page_rank.csv', sep = ',')
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




df = pd.read_csv(f'heuristics_experiment/SDHDN_{model}.csv', sep = ',')
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


print('FINAL - single_discount_high_degree_nodes', hv_MAP/hv_SD)

df = pd.read_csv(f'heuristics_experiment/HDN_{model}.csv', sep = ',')
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
print('FINAL - high_degree_nodes', hv_MAP/hv_HD)

df = pd.read_csv(f'heuristics_experiment/CELF_{model}_25.csv', sep = ',')
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


hv_CELF = metric.do(A) / tot

print(hv_CELF)


print('FINAL - CELF', hv_MAP/hv_CELF)
import matplotlib.pyplot as plt






# plt.scatter(x1,y1, color='yellow', label='Single Discount Heuristic', facecolor='none')
# plt.scatter(x2,y2, color='pink', label='CELF Heuristic')
# plt.scatter(x2,y2, color='grey', label='Highest Degree Heuristic', facecolor='none')

# plt.scatter(x,y, color='black', label='Mapping')


plt.scatter(t2[:,0],t2[:,1], color='purple', label='Single Discount Heuristic', facecolor='none')
plt.scatter(t4[:,0],t4[:,1], color='olive', label='CELF Heuristic',facecolor='none')
plt.scatter(t3[:,0],t3[:,1] , color='grey', label='Highest Degree Heuristic', facecolor='none')

plt.scatter(t1[:,0],t1[:,1],color='black', label='Mapping')
plt.xlim(0,50)
plt.legend()


plt.title('IC MODEL')
plt.savefig('prova.png', format='png')