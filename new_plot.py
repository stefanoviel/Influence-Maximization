import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

filename = "/Users/elia/Desktop/Influence-Maximization/RandomGraph-N200-E8052-population.csv"
df = pd.read_csv(filename, sep=",")
df = df.sort_values(by=['generations', 'n_nodes', 'influence'])

x1 = np.linspace(df['n_nodes'].min(), df['n_nodes'].max(), len(df['n_nodes'].unique()))
y1 = np.linspace(df['generations'].min(), df['generations'].max(), len(df['generations'].unique()))
x2, y2 = np.meshgrid(x1, y1)
from scipy.interpolate import griddata

z2 = griddata((df['n_nodes'], df['generations']), df['influence'], (x2, y2), method='cubic')
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)
from matplotlib.ticker import LinearLocator, FormatStrFormatter
ax.set_zlim(0, df['influence'].max())

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Meshgrid Created from 3 1D Arrays')
# ~~~~ MODIFICATION TO EXAMPLE ENDS HERE ~~~~ #

plt.show()