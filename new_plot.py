import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

filename = "/Users/elia/Desktop/Influence-Maximization/RandomGraph-N200-E5942-population.csv"
df = pd.read_csv(filename, sep=",")
df = df.sort_values(by=['generations', 'n_nodes', 'influence'])

x1 = np.linspace(df['n_nodes'].min(), df['n_nodes'].max(), len(df['n_nodes']))
y1 = np.linspace(df['generations'].min(), df['generations'].max(), len(df['generations']))
x2, y2 = np.meshgrid(x1, y1)
from scipy.interpolate import griddata

z2 = griddata((df['n_nodes'], df['generations']), df['influence'], (x2, y2), method='linear')
from matplotlib import cm

ma = np.nanmax(df['influence'])
norm = matplotlib.colors.Normalize(vmin = 0, vmax = ma, clip = True)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['n_nodes'].to_list(),df['generations'].to_list(),df['influence'].to_list(), alpha=1, color="red")

surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.inferno,
    linewidth=0, antialiased=False, norm=norm, alpha=1)
#ax.set_zlim(-1.01, 1.01)
from matplotlib.ticker import LinearLocator, FormatStrFormatter
ax.set_zlim(0, df['influence'].max())

#ax.zaxis.set_major_locator(LinearLocator(10))
ax.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax.yaxis.set_ticks(np.arange(0, df["generations"].max()+1, 50))
ax.zaxis.set_ticks(np.arange(0, df["influence"].max()+1, 30))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Meshgrid Created from 3 1D Arrays')
# ~~~~ MODIFICATION TO EXAMPLE ENDS HERE ~~~~ #

plt.show()