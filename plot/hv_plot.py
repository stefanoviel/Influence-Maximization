import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '')
from src.load import read_graph


df = pd.read_csv("fb_org-k138-p0.05-IC-NEW_3_OBJ_hv_.csv",sep=",")
df1 = pd.read_csv("fb_org_TRUE-2.0-k68-p0.05-IC-NEW_3_OBJ_hv_.csv",sep=",")

df2 = pd.read_csv("fb_org_TRUE-4.0-k34-p0.05-IC-NEW_3_OBJ_hv_.csv",sep=",")
df4= pd.read_csv("fb_org_TRUE-8.0-k17-p0.05-IC-NEW_3_OBJ_hv_.csv",sep=",")

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(df.generation, df.hv, color="red", label="0riginal")
axs[0,0].plot(df1.generation, df1.hv, color="orange", label="75%")
axs[0,0].plot(df2.generation, df2.hv, color="blue", label="50%")
axs[0,0].plot(df4.generation, df4.hv, color="green", label="25%")
#axs[0,0].xlabel('Generation')
#axs[0,0].ylabel('Hypervolume')
axs[0,0].legend()
axs[0,0].set_title('HV')

plt.show()

axs[0,1].plot(df.generation, df.influence_k, color="red", label="0riginal")
axs[0,1].plot(df1.generation, df1.influence_k, color="orange", label="75%")
axs[0,1].plot(df2.generation, df2.influence_k, color="blue", label="50%")
axs[0,1].plot(df4.generation, df4.influence_k, color="green", label="25%")
#axs[0,1].xlabel('Generation')
#axs[0,1].ylabel('Hypervolume')
axs[0,1].set_title('HV INFLUENCE-K')



axs[1,0].plot(df.generation, df.influence_comm, color="red", label="0riginal")
axs[1,0].plot(df1.generation, df1.influence_comm, color="orange", label="75%")
axs[1,0].plot(df2.generation, df2.influence_comm, color="blue", label="50%")
axs[1,0].plot(df4.generation, df4.influence_comm, color="green", label="25%")
#axs[1,0].xlabel('Generation')
#axs[1,0].ylabel('Hypervolume')
axs[1,0].set_title('HV INFLUENCE-COMM')




axs[1,1].plot(df.generation, df.k_comm, color="red", label="0riginal")
axs[1,1].plot(df1.generation, df1.k_comm, color="orange", label="75%")
axs[1,1].plot(df2.generation, df2.k_comm, color="blue", label="50%")
axs[1,1].plot(df4.generation, df4.k_comm, color="green", label="25%")
#axs[1,1].xlabel('Generation')
#axs[1,1].ylabel('Hypervolume')
axs[1,1].set_title('HV K-COMM')

for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Hypervolume')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig.suptitle('HYPERVOLUME SBM LT MODEL')
plt.show()


