import pandas as pd
import matplotlib.pylab as plt
import sys
import numpy as np
import os
sys.path.insert(0, '')
from src.load import read_graph
filename1 = "scale_results_csv/pgp.csv"
df = pd.read_csv(filename1, sep=",")

x1 = [x for x in range(1,len(df)+1)]
y1 = df["#C"]
y2 = df["min_size"]

name = os.path.basename(filename1)
name = name.replace('.csv','')
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(name)
ax1.plot(x1, y1, color="blue", label='No. Communities' )
ax1.legend()
ax1.set_xlabel('Resolution')
ax1.set_ylabel('No. Communities')
ax2.plot(x1, y2 ,color="orange", label='Min Size Comm' )
ax2.set_xlabel('Resolution')
ax2.set_ylabel('Min Size Comm')
ax2.legend()
plt.show()