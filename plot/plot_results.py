from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 


filename = "/Users/elia/Desktop/final_experiments/amazon0302/Amazon0302-k400-WC.csv"

df = pd.read_csv(filename)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,3))
x = df["n_nodes"]
z = df["influence"]
y = df["n_simulation"]

tit = str(os.path.basename(filename))
tit = tit.replace(".csv", "")
#fig.suptitle(tit)

ax1.scatter(x, z)
ax1.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 50))
ax1.yaxis.set_ticks(np.arange(0,df["influence"].max()+1, 150))
ax1.set_xlabel("Nodes")

ax1.set_ylabel("Influence")

ax2.scatter(y, z)
ax2.xaxis.set_ticks(np.arange(0, df["n_simulation"].max()+1, 1))
ax2.yaxis.set_ticks(np.arange(0,df["influence"].max()+1, 150))
ax2.set_xlabel("Converge Time")

ax2.set_ylabel("Influence")

ax3.scatter(y, x)
ax3.yaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 40))
ax3.xaxis.set_ticks(np.arange(0,df["n_simulation"].max()+1, 1))
ax3.set_xlabel("Converge Time")

ax3.set_ylabel("Nodes")

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
fig.tight_layout()

#plt.xlabel('Fitness')
#plt.ylabel('Generations')

plt.show()