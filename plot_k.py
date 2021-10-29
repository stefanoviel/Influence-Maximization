import matplotlib.pyplot as plt
import pandas as pd

filename = "facebook_combined-k100-p1-LT.csv.csv"
filename1_3 = "facebook_combined_scale_1.33-k75-p1.33-LT.csv.csv"
filename1_5 = "facebook_combined_scale_1.5-k66-p1.5-LT.csv.csv"  
filename2= "facebook_combined_scale_2-k50-p2-LT.csv.csv"
filename3 = "facebook_combined_scale_3-k33-p3-LT.csv.csv"
filename4 = "/Users/elia/Desktop/Influence-Maximization/facebook_combined_scale_4-k25-p4-LT.csv.csv"
filename5 = "/Users/elia/Desktop/Influence-Maximization/facebook_combined_scale_5-k20-p5-LT.csv.csv"

df = pd.read_csv(filename)
x = df["n_nodes"].to_list()
z = df["influence"]
y = df["time"]
df = df.drop_duplicates(subset=['n_nodes', 'time'], keep='last')
df13 = pd.read_csv(filename1_3)

x0 = df13["n_nodes"].to_list()
z0 = df13["influence"]
y0 = df13["time"]
df13 = df13.drop_duplicates(subset=['n_nodes', 'time'], keep='last')

df15 = pd.read_csv(filename1_5)
x1 =df15["n_nodes"].to_list()
z1 =df15["influence"]
y1 =df15["time"]
df15 = df15.drop_duplicates(subset=['n_nodes', 'time'], keep='last')

df2 = pd.read_csv(filename2)
x2 =df2["n_nodes"].to_list()
z2 =df2["influence"]
y2 =df2["time"]
df2 = df2.drop_duplicates(subset=['n_nodes', 'time'], keep='last')

df3 = pd.read_csv(filename3)
x3 =df3["n_nodes"].to_list()
z3 =df3["influence"]
y3 =df3["time"]
df3 = df3.drop_duplicates(subset=['n_nodes', 'time'], keep='last')

df4 = pd.read_csv(filename4)
x4 =df4["n_nodes"].to_list()
z4 =df4["influence"]
y4 =df4["time"]
df4 = df4.drop_duplicates(subset=['n_nodes', 'time'], keep='last')

df5 = pd.read_csv(filename5)
x5 =df5["n_nodes"].to_list()
z5 =df5["influence"]
y5 =df5["time"]
df5 = df5.drop_duplicates(subset=['n_nodes', 'time'], keep='last')



import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

y = [1,2,3,4,5,6,7]
f,((ax1,ax2,ax3,axcb) ,(ax4,ax5,ax6,ax7)) = plt.subplots(2,4, 
            gridspec_kw={'width_ratios':[1,1,1,0.08]})
#ax1.get_shared_y_axes().join(ax2,ax3)

flights = df.pivot("time", "n_nodes", "influence")
g1 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax1,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto',center=0)
g1.set_ylabel('No. Communities')
g1.set_xlabel('No. Nodes')
g1.set_title("Original Graph")
flights = df13.pivot("time", "n_nodes", "influence")
g2 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax2,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
g2.set_ylabel('No. Communities')
g2.set_xlabel('No. Nodes')
g2.set_title("75%")

flights = df15.pivot("time", "n_nodes", "influence")

g3 = sns.heatmap(flights,cmap="YlGnBu",ax=ax3, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
g3.set_ylabel('No. Communities')
g3.set_xlabel('No. Nodes')
g3.set_title("66%")

flights = df2.pivot("time", "n_nodes", "influence")
g4 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax4,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
g4.set_ylabel('No. Communities')
g4.set_xlabel('No. Nodes')
g4.set_title("50%")
flights = df3.pivot("time", "n_nodes", "influence")
g5 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax5,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
g5.set_ylabel('No. Communities')
g5.set_xlabel('No. Nodes')
g5.set_title("33%")
flights = df4.pivot("time", "n_nodes", "influence")
g6 = sns.heatmap(flights,cmap="YlGnBu",ax=ax6, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
g6.set_ylabel('No. Communities')
g6.set_xlabel('No. Nodes')
g6.set_title("25%")
# may be needed to rotate the ticklabels correctly:
for ax in [g1,g2,g3,g5,g5,g6]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
plt.tight_layout()
plt.show()


f,(ax1,ax2,ax3,axcb) = plt.subplots(1,4, 
            gridspec_kw={'width_ratios':[1,1,1,0.08]})
#ax1.get_shared_y_axes().join(ax2,ax3)

flights = df.pivot("time", "n_nodes", "influence")
g1 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax1,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto',center=0)
g1.set_ylabel('No. Communities')
g1.set_xlabel('No. Nodes')
g1.set_title("Original Graph")

flights = df2.pivot("time", "n_nodes", "influence")
g2 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax2,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
g2.set_ylabel('No. Communities')
g2.set_xlabel('No. Nodes')
g2.set_title("50%")


df_fake = pd.DataFrame()
df_fake["n_nodes"] = [round(x/2,1) for x in df["n_nodes"].to_list()]
df_fake["influence"] = [x/2 for x in df["influence"].to_list()]
df_fake["time"] = [x for x in df["time"].to_list()]

flights = df_fake.pivot("time", "n_nodes", "influence")

g3 = sns.heatmap(flights,cmap="YlGnBu",ax=ax3, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
g3.set_ylabel('No. Communities')
g3.set_xlabel('No. No')
g3.set_title("Ideal Influence%")


plt.tight_layout()

plt.show()