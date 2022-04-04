from ast import Break
import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv('experiments/fb-pages-artist_32-IC-36-0.01/run-1_hv_.csv')
print(df)

y = df["hv"].to_list()
x = [z+1 for z in range(len(y))]



N = 200
max_value = None
X = None
for i in range(len(y)):
    if i >=N:
        if y[i-N] +(y[i-N]*0.01) >= y[i] :
            print('Finaò ', i)
            print(y[i-N],(y[i-N] +(y[i-N]*0.01)),y[i])
            X = [i,i]
            break
        else:
            print((y[i-N] +(y[i-N]*0.01)) - y[i])
    if X != None:
        break
plt.plot(x,y)

Y = []
Y.append(min(y))
Y.append(max(y))
plt.plot(X,Y,color='red')
plt.show()