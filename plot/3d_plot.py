import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from matplotlib import animation


def init():

    #surf = ax.plot_trisurf(x1,y1,z1, cmap='viridis_r', linewidth=0,alpha = 0.99, edgecolor = 'k', norm=norm)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig,


filename = "C:/Users/edosc/OneDrive/Desktop/UniTn/Second Semester/Bio-Inspired Artificial Intelligence/Influence-Maximization/Influence-Maximization/Facebook-WC-Graph-N4039-E88234-population.csv"
df = pd.read_csv(filename, sep=",")
df = df.sort_values(by=['n_nodes', 'n_simulation', 'influence'])
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

print('Lenght Dataset (i.e number of elements) {}'.format(len(df)))
#df = df.sort_values(by="n_nodes")
x1= df["n_nodes"]
y1 = df["n_simulation"]
z1 = df["influence"]
ax.scatter(x1,y1,z1, alpha=1, color="red")



# x = np.linspace(df["n_nodes"].min(), df["n_nodes"].max(), 50)
# y = np.linspace(df["n_simulation"].min(), df["n_simulation"].max(), 50) 
# z = np.linspace(df["influence"].min(), df["influence"].max(),  50)

print(df["influence"])

ax.set_title("Influence Maximization")
ax.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 1))
ax.yaxis.set_ticks(np.arange(0,df["n_simulation"].max()+1, 1))
ax.zaxis.set_ticks(np.arange(0, df["influence"].max()+1, 30))

ax.set_xlabel("Nodes")

ax.set_zlabel("Influence")

ax.set_ylabel("Converge Time")

dfnew = pd.DataFrame(index=y1.values, columns=x1.values)

for i in range(len(x1)):
    print(i)
    print(z1[i])
    dfnew.iloc[i,i] = z1[i]

dfnew = dfnew.fillna(0)
#print(dfnew)


#print("ciao")
xv, yv = np.meshgrid(dfnew.columns, dfnew.index)
ma = np.nanmax(dfnew.values)
norm = matplotlib.colors.Normalize(vmin = 0, vmax = df["influence"].max(), clip = True)

surf = ax.plot_trisurf(x1,y1,z1, cmap='viridis_r', linewidth=0,alpha = 0.99, edgecolor = 'k', norm=norm)
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.savefig('{}.png'.format(filename))
plt.show()

def animate(i):
    # azimuth angle : 0 deg to 360 deg
    ax.view_init(elev=10, azim=i*4)
    return fig,

# Animate
def save_video():
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=90, interval=50, blit=True)

    fn = "ca-GrQcGraph-Mean-scatter"
    ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
    ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)

#save_video()