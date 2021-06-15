import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')
df = pd.read_csv("2021-06-15-21-26-18-population.csv", sep=",")

print(df)
x1= df["n_nodes"]
y1 = df["influence"]
z1 = df["generations"]
x1 = x1.sort_values()
y1 = y1.sort_values()
z1 = z1.sort_values()
x = np.linspace(df["n_nodes"].min(), df["n_nodes"].max(), 50)
y = np.linspace(df["influence"].min(), df["influence"].max(),  50)
z = np.linspace(df["generations"].min(), df["generations"].max(), 50)


ax.plot(x,y,z, color="red") # plot the point (2,3,4) on the figure
ax.scatter(x1,y1,z1)
ax.set_title("Influence Maximization")


ax.set_xlabel("Nodes")

ax.set_ylabel("Influence")

ax.set_zlabel("Time/Generations")
plt.show()
