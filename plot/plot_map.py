import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation
import sys
sys.path.insert(0, '')
from src.load import read_graph

G_original = read_graph('graphs/pgp.txt')
k = G_original.number_of_nodes() * 0.025
comm_original = pd.read_csv('comm_ground_truth/pgp.csv')
comm_original = max(set(comm_original["comm"].to_list()))




filename = "experiments/pgp_8-WC/run-1.csv"
df = pd.read_csv(filename, sep=",")
x=  df["n_nodes"].to_list()
y = df["communities"]
z = df["influence"]

filename = "experiments/pgp-WC/run-1.csv"
df = pd.read_csv(filename, sep=",")

x0= df["n_nodes"].to_list()
y0 = df["communities"].to_list()
z0 = df["influence"].to_list()


pf = []
for i in range(len(x0)):
    pf.append([-z0[i],-(2.5 - x0[i]), -y0[i]])

filename = "core_8_PGP_true.csv"




df = pd.read_csv(filename, sep=",")

x1= df["n_nodes"]
y1 = df["communities"]
z1 = df["influence"]



A = []
for i in range(len(x1)):
    A.append([-z1[i],-(2.5 - x1[i]), -y1[i]])




fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,6))

ax1.scatter(z,x,color="green")
ax1.scatter(z0,x0,color="red")
ax1.scatter(z1,x1,color="black")
#ax1.title('Facebook Politicians IC p=0.01 model')
ax1.set_xlabel('% Influenced Nodes')
ax1.set_ylabel('% Nodes as seed set')
#ax1.legend()
#ax1.set_xlim(0,100)
ax1.set_ylim(0,2.5)

#ax1.savefig('aa')
#ax1.show()
#ax1.cla()



ax2.scatter(y,x,color="green")
ax2.scatter(y0,x0,color="red",)
ax2.scatter(y1,x1,color="black")
#ax2.title('Facebook Politicians IC model')
ax2.set_xlabel('Communities')
ax2.set_ylabel('% Nodes as seed set')
ax1.set_ylim(0,2.5)
#ax2.set_xlim(1,max(y0))
#ax2.legend()
#ax2.show()
#ax2.savefig('aa_a')

#plt.cla()

ax3.scatter(z,y,color="green",label="1/2")
ax3.scatter(z0,y0,color="red", label="Original")
ax3.scatter(z1,y1,color="black",label="MAP")
#ax3.title('Facebook Politicians IC model')
ax3.set_xlabel('% Influenced Nodes')
ax3.set_ylabel('Communities')
#ax3.legend()
#ax3.set_xlim(0,100)
#ax3.savefig('aa_aa')
#plt.legend(loc='center left')

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.85)  
plt.title('Degree Centrality')
plt.savefig('k_core_8')
plt.show()
#plt.cla()




from pymoo.factory import get_performance_indicator
pf = np.array(pf)
A = np.array(A)
gd = get_performance_indicator("gd", pf)
print("GD", gd.do(A))



tot = 100 * 2.5 * (comm_original-1)
from pymoo.indicators.hv import Hypervolume

metric = Hypervolume(ref_point= np.array([0,0,-1]),
                    norm_ref_point=False,
                    zero_to_one=False)

hv_original = metric.do(pf) /tot

hv_MAP = metric.do(A) / tot


print((hv_MAP/hv_original))
print(hv_original, hv_MAP)
#print(hv_original - hv_MAP)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(z, y, x, marker='o', color="green")
ax.scatter(z0, y0, x0, marker='o', color="red")
ax.scatter(z1, y1, x1, marker='o', color="black")
plt.savefig('a3d')
plt.show()
exit(0)