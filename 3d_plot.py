import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

filename = "D:/Users/edosc/PycharmProjects/Influence-Maximization/RandomGraph-N200-E7833-population.csv"
df = pd.read_csv(filename, sep=",")
fig = plt.figure(figsize=(8, 8))
fig1 = plt.figure(figsize=(8, 8))
fig2 = plt.figure(figsize=(8, 8))
fig3 = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection='3d')
ax1 = fig1.add_subplot(111, projection="3d")
ax2 = fig2.add_subplot(111, projection="3d")
ax3 = fig3.add_subplot(111, projection="3d")
# df = df.sort_values(by="n_nodes")
x1 = df["n_nodes"]
y1 = df["influence"]
z1 = df["generations"]


# x = np.linspace(df["n_nodes"].min(), df["n_nodes"].max(), 50)
# y = np.linspace(df["influence"].min(), df["influence"].max(),  50)
# z = np.linspace(df["generations"].min(), df["generations"].max(), 50)

# ax.plot(x,y,z, color="red") # plot the point (2,3,4) on the figure
ax.scatter(x1, y1, z1, alpha=1)
ax1.scatter(x1, y1, alpha=1)  # Nodes and influence
ax2.scatter(x1, z1, alpha=1)  # Nodes and generations
ax3.scatter(y1, z1, alpha=1)  # Influence and generations

n_nodes_influence_generations = df.sort_values(["n_nodes", "influence", "generations"], ascending=[False, True, False])
n_nodes_influence = df.sort_values(['n_nodes', 'influence'], ascending=[False, True])
n_nodes_generations = df.sort_values(['n_nodes', 'generations'], ascending=[False, False])
influence_generations = df.sort_values(["influence", "generations"], ascending=[True, False])

# x1 = x1.sort_values()

# y1 = y1.sort_values()  # Modificare qua


# z1 = z1.sort_values()  # Modificare qua e fare in modo che ordini secondo due variabili

ax.plot(n_nodes_influence_generations["n_nodes"], n_nodes_influence_generations["influence"], n_nodes_influence_generations["generations"], color="red")
ax1.plot(n_nodes_influence["n_nodes"], n_nodes_influence["influence"], color="orange")
ax2.plot(n_nodes_generations["n_nodes"], n_nodes_generations["generations"], color="green")
ax3.plot(influence_generations["influence"], influence_generations["generations"], color="black")


ax.set_title("Influence Maximization")
ax.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax.yaxis.set_ticks(np.arange(0, df["influence"].max()+1, 2))
ax.zaxis.set_ticks(np.arange(0, df["generations"].max()+1, 5))
ax.set_xlabel("Nodes")
ax.set_ylabel("Influence")
ax.set_zlabel("Time/Generations")

ax1.set_title("Influence Maximization")
ax1.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax1.yaxis.set_ticks(np.arange(0, df["influence"].max()+1, 2))
ax1.set_xlabel("Nodes")
ax1.set_ylabel("Influence")

ax2.set_title("Influence Maximization")
ax2.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax2.yaxis.set_ticks(np.arange(0, df["generations"].max()+1, 5))
ax2.set_xlabel("Nodes")
ax2.set_ylabel("Generations")

ax3.set_title("Influence Maximization")
ax3.xaxis.set_ticks(np.arange(0, df["influence"].max()+1, 2))
ax3.yaxis.set_ticks(np.arange(0, df["generations"].max()+1, 5))
ax3.set_xlabel("Influence")
ax3.set_ylabel("Generations")

plt.savefig('{}.png'.format(filename))
plt.show()
