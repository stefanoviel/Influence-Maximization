import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '')
from src.load import read_graph


df = pd.read_csv("graph_SBM_small-k100-p0.1-IC-best_hv_hv_.csv",sep=",")
df2 = pd.read_csv("graph_SBM_small_TRUE-2-k50-p0.1-IC-best_hv_hv_.csv",sep=",")
df4= pd.read_csv("graph_SBM_small_TRUE-4-k25-p0.1-IC-best_hv_hv_.csv",sep=",")


plt.plot(df.generation, df.hv, color="red", label="0riginal")
plt.plot(df2.generation, df2.hv, color="blue", label="50%")
plt.plot(df4.generation, df4.hv, color="green", label="25%")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.legend()
plt.show()

