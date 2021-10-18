import pandas as pd
import matplotlib.pylab as plt
import sys
import numpy as np

sys.path.insert(0, '')
from src.load import read_graph
df = pd.read_csv("/Users/elia/Desktop/Influence-Maximization/twitter.csv", sep=",")

g =read_graph(filename="graphs/ego-twitter.txt")
original_density = (2*g.number_of_edges())/ (g.number_of_nodes()*(g.number_of_nodes()-1))
print("Density --> {0}".format(original_density)) 
print(df)

x1 = [x for x in range(1,len(df)+1)]
y1 = df["#C"]

plt.plot(x1, y1, label = "communities", color="green")
plt.xlabel('r - resolution')
plt.ylabel('#C -  no communities')
plt.legend()
#plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear f
plt.close()


y2 = df["#min_size"]
plt.plot(x1, y2, label = "size", color="blue")
plt.xlabel('r - resolution')
plt.ylabel('#S - Size smallest community')
plt.legend()
plt.yticks(np.arange(0, np.mean(y2)*2, step=1))
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear f
plt.close()



y3 = df["density"].to_list()

y3 = [float(x) for x in y3]
density = []
for item in x1:
    density.append(original_density)

plt.plot(x1, y3, label = "density", color="black")
plt.plot(x1,density, label="original density", color="red")
plt.xlabel('r - resolution')
plt.ylabel('#d - density value')
plt.legend()
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear f
plt.close()



y3.append(original_density)
y3 = (y3 - np.min(y3))/np.ptp(y3)
original_density = y3[len(y3)-1]
y3 = np.delete(y3, len(y3)-1)
y1 = (y1 - np.min(y1))/np.ptp(y1)
density = []
for item in x1:
    density.append(original_density)

plt.plot(x1, y1, label = "communities", color="green")
plt.plot(x1, y3, label = "size", color="black")
plt.plot(x1,density, label="original density", color="red")

plt.xlabel('r - resolution')

# Set the y axis label of the current axis.
plt.ylabel('S - Size')
plt.legend()
plt.xticks(np.arange(0, len(x1), step=2))
plt.show()

plt.cla()   # Clear axis
plt.clf()   # Clear f
plt.close()
