import pandas as pd
import matplotlib.pylab as plt
import sys
import numpy as np
import os
sys.path.insert(0, '')
from src.load import read_graph
filename1 = "scale_results_csv/facebook_combined_5.csv"
df = pd.read_csv(filename1, sep=",")
filename="graphs/facebook_combined.txt"
g =read_graph(filename)
name = os.path.basename(filename1)
original_density = (2*g.number_of_edges())/ (g.number_of_nodes()*(g.number_of_nodes()-1))
print("Density --> {0}".format(original_density)) 

x1 = [x for x in range(1,len(df)+1)]
y1 = df["#C"]

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(11,7)
name = name.replace(".csv","")
fig.suptitle(name, fontsize=16)

axs[0, 0].plot(x1, y1, label = "communities", color="green")
#axs[0, 0].xlabel('r - resolution')
#axs[0, 0].ylabel('#C -  no communities')
axs[0, 0].legend()



y2 = df["#min_size"].to_list()
axs[0, 1].plot(x1, y2, label = "size", color="blue")
#axs[0, 1].xlabel('r - resolution')
#axs[0, 1].ylabel('#S - Size smallest community')
axs[0, 1].legend()
#axs[0, 1].yticks(np.arange(0, np.mean(y2)*2, step=1))




y3 = df["density"].to_list()

y3 = [float(x) for x in y3]
density = []
for item in x1:
    density.append(original_density)

axs[1, 0].plot(x1, y3, label = "density", color="black")
axs[1, 0].plot(x1,density, label="original density", color="red")
#axs[1, 0].xlabel('r - resolution')
#axs[1, 0].ylabel('#d - density value')
axs[1, 0].legend()



y3.append(original_density)
y3 = (y3 - np.min(y3))/np.ptp(y3)
original_density = y3[len(y3)-1]
y3 = np.delete(y3, len(y3)-1)
y1 = (y1 - np.min(y1))/np.ptp(y1)
print(y2)
y2 = (y2 - np.min(y2))/np.ptp(y2)
print(y2)
density = []
for item in x1:
    density.append(original_density)

axs[1, 1].plot(x1, y1, label = "communities", color="green")
axs[1, 1].plot(x1, y3, label = "density", color="black")
axs[1, 1].plot(x1,density, label="original density", color="red")
axs[1, 1].plot(x1,y2, label="size", color="blue")

#axs[1, 1].xlabel('r - resolution')

# Set the y axis label of the current axis.
#axs[1, 1].ylabel('S - Size')
#axs[1, 1].xticks(np.arange(0, len(x1), step=2))

xlabel = ['r - resolution','r - resolution','r - resolution','r - resolution']
ylabel = ['C -  no communities','#S - Size smallest community','#d - density value', '']
i = 0
for ax in axs.flat:
    ax.set(xlabel=xlabel[i], ylabel=ylabel[i])
    i +=1

#plt.savefig("prova.png",figsize=(40, 10),dpi=1000)
plt.show()
