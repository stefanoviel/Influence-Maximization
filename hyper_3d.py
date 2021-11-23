import pandas as pd
import matplotlib.pyplot as plt

best = pd.read_csv('best.csv',sep=",")
worse = pd.read_csv('worse.csv',sep=",")


b1 = best["influence"]
b2 = best["nodes"]
b3 = best["comm"]

w1 = worse["influence"]
w2 = worse["nodes"]
w3 = worse["comm"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(b1,b2,b3, color='red')
ax.scatter(w1,w2,w3, color='green')
plt.show()