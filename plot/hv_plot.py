import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '')
from src.load import read_graph


df = pd.read_csv("graph_SBM_small-k100-p0.05-IC-best_hv_hv_.csv",sep=",")
df2 = pd.read_csv("graph_SBM_small_TRUE-2-k50-p0.05-IC-best_hv_hv_.csv",sep=",")
df4= pd.read_csv("graph_SBM_small_TRUE-4-k25-p0.05-IC-best_hv_hv_.csv",sep=",")


plt.plot(df.generation, df.hv, color="red", label="0riginal")
plt.plot(df2.generation, df2.hv, color="blue", label="50%")
plt.plot(df4.generation, df4.hv, color="green", label="25%")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.legend()
plt.title('HV')
plt.show()
plt.cla()
plt.close()

plt.plot(df.generation, df.influence_k, color="red", label="0riginal")
plt.plot(df2.generation, df2.influence_k, color="blue", label="50%")
plt.plot(df4.generation, df4.influence_k, color="green", label="25%")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.title('HV INFLUENCE-K')

plt.legend()
plt.show()


plt.plot(df.generation, df.influence_comm, color="red", label="0riginal")
plt.plot(df2.generation, df2.influence_comm, color="blue", label="50%")
plt.plot(df4.generation, df4.influence_comm, color="green", label="25%")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.title('HV INFLUENCE-COMM')

plt.legend()
plt.show()


plt.plot(df.generation, df.k_comm, color="red", label="0riginal")
plt.plot(df2.generation, df2.k_comm, color="blue", label="50%")
plt.plot(df4.generation, df4.k_comm, color="green", label="25%")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.title('HV K-COMM')
plt.legend()
plt.show()


