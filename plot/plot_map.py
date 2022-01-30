import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation
import sys
sys.path.insert(0, '')
from src.load import read_graph


    
filename = "experiments/CA-GrQc_2-IC/run-1.csv"

df = pd.read_csv(filename, sep=",")


x=  df["n_nodes"].to_list()
y = df["communities"]
z = df["influence"]


filename = "experiments/CA-GrQc-IC/run-1.csv"

df = pd.read_csv(filename, sep=",")

x0= df["n_nodes"].to_list()
y0 = df["communities"].to_list()
z0 = df["influence"].to_list()



filename = "map_degree_comm.csv"


df = pd.read_csv(filename, sep=",")

x1= df["n_nodes"]
y1 = df["communities"]
z1 = df["influence"]

print(df)


#plot(x1,y1,z1)




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
#ax2.set_xlim(1,max(y2))
#ax2.legend()
#ax2.show()
#ax2.savefig('aa_a')

#plt.cla()

ax3.scatter(z,y,color="green",label="1/8")
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
plt.savefig('CIAO')
plt.show()
#plt.cla()
exit(0)