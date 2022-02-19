import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation
import sys
sys.path.insert(0, '')
from src.load import read_graph

G_original = read_graph('graphs/fb-pages-public-figure.txt')
k = G_original.number_of_nodes() * 0.025
comm_original = pd.read_csv('comm_ground_truth/fb-pages-public-figure.csv')
comm_original = max(set(comm_original["comm"].to_list()))




filename = "experiments/fb-pages-public-figure_2-WC/run-1.csv"
df = pd.read_csv(filename, sep=",")
x=  df["n_nodes"].to_list()
z = df["influence"]

filename = "experiments/fb-pages-public-figure-WC/run-1.csv"
df = pd.read_csv(filename, sep=",")

x0= df["n_nodes"].to_list()
z0 = df["influence"].to_list()


pf = []
pf_G = []
for i in range(len(x0)):
    pf.append([-z0[i],-(2.5 - x0[i])])
    pf_G.append([z0[i], x0[i]])
filename = "b-pages-public-figure_WC_2.csv"




df = pd.read_csv(filename, sep=",")

x1= df["n_nodes"]
y1 = df["communities"]
z1 = df["influence"]



A = []
A_G = []
for i in range(len(x1)):
    A.append([-z1[i],-(2.5 - x1[i])])
    A_G.append([z1[i], x1[i]])


plt.scatter(z,x,color="green",label='8')
plt.scatter(z0,x0,color="red", label='Original')
plt.scatter(z1,x1,color="black", label='Map')
plt.xlabel('% Influenced Nodes')
plt.ylabel('% Nodes as seed set')
plt.legend()
#plt.xlim(0,100)
plt.ylim(0,2.5)

#plt.savefig('aa')
plt.show()




from pymoo.factory import get_performance_indicator
pf = np.array(pf)
A = np.array(A)
pf_G = np.array(pf_G)
A_G = np.array(A_G)
gd = get_performance_indicator("gd", pf_G)
print("GD", gd.do(A_G))



tot = 100 * 2.5 
from pymoo.indicators.hv import Hypervolume

metric = Hypervolume(ref_point= np.array([0,0]),
                    norm_ref_point=False,
                    zero_to_one=False)

hv_original = metric.do(pf) /tot

hv_MAP = metric.do(A) / tot


print((hv_MAP/hv_original))
print(hv_original, hv_MAP)
#print(hv_original - hv_MAP)
