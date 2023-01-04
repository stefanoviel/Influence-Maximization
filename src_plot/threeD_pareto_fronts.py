# iterate through the runs, find the one with the best hypervolume
# compute the graph for that one

import sys
import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np

print(os.listdir())

dir = sys.argv[1]
max_hv = 0
for i in range(1, 11): 
    hv = pd.read_csv(f'{dir}/run-{i}_hv_.csv', sep = ',')
    if hv["hv"].max() > max_hv: 
        max_hv = hv["hv"].max()
        df = pd.read_csv(f'{dir}/run-{i}.csv', sep = ',')
        n_nodes = df["n_nodes"]
        communities = df["communities"]
        influence = df["influence"]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(n_nodes, communities, influence)
plt.show()
