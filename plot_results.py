from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/elia/Desktop/Influence-Maximization-master/2021-06-14-20-52-31-population.csv")

print(df)
print(df.columns)
df = df.groupby(by='n_nodes')['influence'].mean().reset_index()
y = df["n_nodes"]
x = df["influence"]
x = x.sort_values()
y = y.sort_values()
plt.plot(x, y, marker="o")
plt.xlabel('Fitness')
plt.ylabel('Number of Nodes')

plt.show()