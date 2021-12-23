import pandas as pd
import matplotlib.pylab as plt
import sys
import numpy as np
import os
sys.path.insert(0, '')
from src.load import read_graph
filename1 = "scale_results_csv/deezerEU.csv"
df = pd.read_csv(filename1, sep=",")

x1 = [x for x in range(1,len(df)+1)]
y1 = df["#C"]
y2 = df["min_size"]


plt.plot(x1,y1)
plt.plot(x1,y2)
plt.show()
