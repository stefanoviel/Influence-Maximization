import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '')
from src.load import read_graph


df4 = pd.read_csv("fb_org_TRUE-8.0-k17-p0-WC-NEW_3_OBJ_hv_.csv",sep=",")
df2 = pd.read_csv("fb_org_TRUE-4.0-k34-p0-WC-NEW_3_OBJ_hv_.csv",sep=",")

df1 = pd.read_csv("fb_org_TRUE-2.0-k68-p0-WC-NEW_3_OBJ_hv_.csv",sep=",")
df= pd.read_csv("prova_WC_hv_.csv",sep=",")

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(df.generation, df.hv, color="red")
axs[0,0].plot(df1.generation, df1.hv, color="orange")
axs[0,0].plot(df2.generation, df2.hv, color="blue")
axs[0,0].plot(df4.generation, df4.hv, color="green")
axs[0,0].set_title('HV')



axs[0,1].plot(df.generation, df.influence_k, color="red")
axs[0,1].plot(df1.generation, df1.influence_k, color="orange")
axs[0,1].plot(df2.generation, df2.influence_k, color="blue")
axs[0,1].plot(df4.generation, df4.influence_k, color="green")
#axs[0,1].xlabel('Generation')
#axs[0,1].ylabel('Hypervolume')
axs[0,1].set_title('HV INFLUENCE-K')



axs[1,0].plot(df.generation, df.influence_comm, color="red")
axs[1,0].plot(df1.generation, df1.influence_comm, color="orange")
axs[1,0].plot(df2.generation, df2.influence_comm, color="blue")
axs[1,0].plot(df4.generation, df4.influence_comm, color="green") 
#axs[1,0].xlabel('Generation')
#axs[1,0].ylabel('Hypervolume')
axs[1,0].set_title('HV INFLUENCE-COMM')




axs[1,1].plot(df.generation, df.k_comm, color="red", label="0riginal")
axs[1,1].plot(df1.generation, df1.k_comm, color="orange", label="1/2")
axs[1,1].plot(df2.generation, df2.k_comm, color="blue", label="1/4")
axs[1,1].plot(df4.generation, df4.k_comm, color="green", label="1/8")
#axs[1,1].xlabel('Generation')
#axs[1,1].ylabel('Hypervolume')
axs[1,1].set_title('HV K-COMM')


#plt.show()
for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Hypervolume')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()



fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.85)  
plt.show()


