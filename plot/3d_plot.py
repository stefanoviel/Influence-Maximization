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


filename = "/Users/elia/Desktop/final_experiments/amazon0302/amazon0302-k200-WC.csv"
df = pd.read_csv(filename, sep=",")
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
print('Lenght Dataset (i.e number of elements) {}'.format(len(df)))
#df = df.sort_values(by="n_nodes")
x1= df["n_nodes"]
y1 = df["n_simulation"]
z1 = df["influence"]
value = []
ax.scatter(x1,y1,z1, alpha=1, color="red")



# x = np.linspace(df["n_nodes"].min(), df["n_nodes"].max(), 50)
# y = np.linspace(df["generations"].min(), df["generations"].max(), 50) 
# z = np.linspace(df["influence"].min(), df["influence"].max(),  50)

print(df["influence"])

ax.set_title(filename)
ax.xaxis.set_ticks(np.arange(0, df["n_nodes"].max()+1, 40))
ax.yaxis.set_ticks(np.arange(0,df["n_simulation"].max()+1, 1))
ax.zaxis.set_ticks(np.arange(0, df["influence"].max()+1, 50))

ax.set_xlabel("Nodes")

ax.set_zlabel("Influence")

ax.set_ylabel("Converge Time")



#plt.savefig('{}.png'.format(filename))
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
    #ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
    ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)

#save_video()