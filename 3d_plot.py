import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

filename = "/Users/elia/Desktop/Influence-Maximization/RandomGraph-N300-E17825-population.csv"
df = pd.read_csv(filename, sep=",")
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

#df = df.sort_values(by="n_nodes")
x1= df["n_nodes"]
y1 = df["influence"]
z1 = df["generations"]



x = np.linspace(df["n_nodes"].min(), df["n_nodes"].max(), 50)
y = np.linspace(df["influence"].min(), df["influence"].max(),  50)
z = np.linspace(df["generations"].min(), df["generations"].max(), 50)


#ax.plot(x,y,z, color="red") # plot the point (2,3,4) on the figure
ax.scatter(x1,y1,z1, alpha=1)
x1 = x1.sort_values()

y1 = y1.sort_values()


z1 = z1.sort_values()
ax.plot(x1, y1, z1, color="red")

ax.set_title("Influence Maximization")
ax.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax.yaxis.set_ticks(np.arange(0, df["influence"].max()+1, 2))
ax.zaxis.set_ticks(np.arange(0, df["generations"].max()+1, 5))

ax.set_xlabel("Nodes")

ax.set_ylabel("Influence")

ax.set_zlabel("Time/Generations")
plt.savefig('{}.png'.format(filename))
plt.show()
