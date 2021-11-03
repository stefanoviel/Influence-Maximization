import matplotlib.pyplot as plt
import pandas as pd

'''
filename = "graph_SBM_small-k100-p0.05-IC.csv.csv"
filename1_3 = "graph_SBM_small_scale_1.33-k75-p0.0665-IC.csv.csv"
filename1_5 = "/Users/elia/Desktop/Influence-Maximization/graph_SBM_small_scale_1.5-k66-p0.07500000000000001-IC.csv.csv"  
filename2= "/Users/elia/Desktop/Influence-Maximization/graph_SBM_small_scale_2-k50-p0.1-IC.csv.csv"
filename3 = "/Users/elia/Desktop/Influence-Maximization/graph_SBM_small_scale_3-k33-p0.15000000000000002-IC.csv.csv"
filename4 = "/Users/elia/Desktop/Influence-Maximization/graph_SBM_small_scale_4-k25-p0.2-IC.csv.csv"
filename5 = "/Users/elia/Desktop/Influence-Maximization/graph_SBM_small_scale_5-k20-p0.25-IC.csv.csv"
'''

# filename = "facebook_combined-k100-p0.05-IC.csv.csv"
# filename1_3 = "facebook_combined_scale_1.33-k75-p0.0665-IC.csv.csv"
# filename1_5 = "facebook_combined_scale_1.5-k66-p0.07500000000000001-IC.csv.csv"  
# filename2= "facebook_combined_scale_2-k50-p0.1-IC.csv.csv"
# filename3 = "facebook_combined_scale_3-k33-p0.15000000000000002-IC.csv.csv"
# filename4 = "facebook_combined_scale_4-k25-p0.2-IC.csv.csv"
# filename5 = "facebook_combined_scale_5-k20-p0.25-IC.csv.csv"
# filename = "/Users/elia/Desktop/Influence-Maximization/facebook_combined-k100-p0.02288800235736791-LT.csv.csv"
# filename1_3 = "facebook_combined_scale_1.33-k75-p0.030493688188469887-LT.csv.csv"
# filename1_5 = "facebook_combined_scale_1.5-k66-p0.03436781609195402-LT.csv.csv"  
# filename2= "facebook_combined_scale_2-k50-p0.0459030556566024-LT.csv.csv"
# filename3 = "facebook_combined_scale_3-k33-p0.06850433894844309-LT.csv.csv"
# filename4 = "facebook_combined_scale_4-k25-p0.091599560761347-LT.csv.csv"
# filename5 = "facebook_combined_scale_5-k20-p0.11329348135496727-LT.csv.csv"

# filename = "graph_SBM_small-k100-p0.05-IC.csv.csv"
# filename1_3 = "graph_SBM_small_scale_1.33-k75-p0.0665-IC.csv.csv"
# filename1_5 = "graph_SBM_small_scale_1.5-k66-p0.07500000000000001-IC.csv.csv"  
# filename2= "graph_SBM_small_scale_2-k50-p0.1-IC.csv.csv"
# filename3 = "graph_SBM_small_scale_3-k33-p0.15000000000000002-IC.csv.csv"
# filename4 = "graph_SBM_small_scale_4-k25-p0.2-IC.csv.csv"
# filename5 = "graph_SBM_small_scale_5-k20-p0.25-IC.csv.csv"

filename = "graph_SBM_small-k100-p0.03603869088891971-LT.csv.csv"
filename1_3 = "graph_SBM_small_scale_1.33-k75-p0.047874953936862794-LT.csv.csv"
filename1_5 = "graph_SBM_small_scale_1.5-k66-p0.0541089999218078-LT.csv.csv"  
filename2= "graph_SBM_small_scale_2-k50-p0.0722841225626741-LT.csv.csv"
filename3 = "graph_SBM_small_scale_3-k33-p0.10850393700787402-LT.csv.csv"
filename4 = "graph_SBM_small_scale_4-k25-p0.14446318156267565-LT.csv.csv"
filename5 = "graph_SBM_small_scale_5-k20-p0.176038062283737-LT.csv.csv"

t = []
df = pd.read_csv(filename)
x = df["n_nodes"].to_list()
z = df["influence"].to_list()

for item in x:
    t.append(1)


df13 = pd.read_csv(filename1_3)

x0 = df13["n_nodes"].to_list()
z0 = df13["influence"].to_list()
for item in x0:
    t.append(1.33)

df15 = pd.read_csv(filename1_5)
x1 =df15["n_nodes"].to_list()
z1 =df15["influence"].to_list()
for item in x1:
    t.append(1.5)

df2 = pd.read_csv(filename2)
x2 =df2["n_nodes"].to_list()
z2 =df2["influence"].to_list()
for item in x2:
    t.append(2)

df3 = pd.read_csv(filename3)
x3 =df3["n_nodes"].to_list()
z3 =df3["influence"].to_list()
for item in x3:
    t.append(3)

df4 = pd.read_csv(filename4)
x4 =df4["n_nodes"].to_list()
z4 =df4["influence"].to_list()

for item in x4:
    t.append(4)

df5 = pd.read_csv(filename5)
x5 =df5["n_nodes"].to_list()
z5 =df5["influence"].to_list()
for item in x5:
    t.append(5)


x_final = [*x, *x0, *x1, *x2, *x3,*x4,*x5]
z_final = [*z, *z0, *z1, *z2, *z3,*z4,*z5]

df = pd.DataFrame()
df["nodes"] = [round(t,2) for t in x_final]
df["influence"] = z_final
df["scale"] = t

print(df)
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

flights = df.pivot("scale", "nodes", "influence")
ax = sns.heatmap(flights,cmap="YlGnBu",linewidths=.5, square=False,annot=False, vmin=0, vmax=max(z))
plt.show()
plt.cla()




plt.scatter(x, z,color="red", label="original")
plt.scatter(x0, z0,color="orange", label="scale 75")
plt.scatter(x1, z1,color="blue", label="scale 66")
plt.scatter(x2, z2,color="grey", label="50")
plt.scatter(x3, z3,color="purple", label="33")
plt.scatter(x4, z4,color="brown", label="25")
plt.scatter(x5, z5,color="pink", label="20")

plt.legend()
plt.show()
# y = [1,2,3,4,5,6,7]
# f,((ax1,ax2,ax3,axcb) ,(ax4,ax5,ax6,ax7)) = plt.subplots(2,4, 
#             gridspec_kw={'width_ratios':[1,1,1,0.08]})
# #ax1.get_shared_y_axes().join(ax2,ax3)

# flights = df.pivot("time", "n_nodes", "influence")
# g1 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax1,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto',center=0)
# g1.set_ylabel('No. Communities')
# g1.set_xlabel('No. Nodes')
# g1.set_title("Original Graph")
# flights = df13.pivot("time", "n_nodes", "influence")
# g2 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax2,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
# g2.set_ylabel('No. Communities')
# g2.set_xlabel('No. Nodes')
# g2.set_title("75%")

# flights = df15.pivot("time", "n_nodes", "influence")

# g3 = sns.heatmap(flights,cmap="YlGnBu",ax=ax3, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
# g3.set_ylabel('No. Communities')
# g3.set_xlabel('No. Nodes')
# g3.set_title("66%")

# flights = df2.pivot("time", "n_nodes", "influence")
# g4 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax4,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
# g4.set_ylabel('No. Communities')
# g4.set_xlabel('No. Nodes')
# g4.set_title("50%")
# flights = df3.pivot("time", "n_nodes", "influence")
# g5 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax5,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
# g5.set_ylabel('No. Communities')
# g5.set_xlabel('No. Nodes')
# g5.set_title("33%")
# flights = df4.pivot("time", "n_nodes", "influence")
# g6 = sns.heatmap(flights,cmap="YlGnBu",ax=ax6, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
# g6.set_ylabel('No. Communities')
# g6.set_xlabel('No. Nodes')
# g6.set_title("25%")
# # may be needed to rotate the ticklabels correctly:
# for ax in [g1,g2,g3,g5,g5,g6]:
#     tl = ax.get_xticklabels()
#     ax.set_xticklabels(tl, rotation=90)
#     tly = ax.get_yticklabels()
#     ax.set_yticklabels(tly, rotation=0)
# plt.tight_layout()
# plt.show()


# f,(ax1,ax2,ax3,axcb) = plt.subplots(1,4, 
#             gridspec_kw={'width_ratios':[1,1,1,0.08]})
# #ax1.get_shared_y_axes().join(ax2,ax3)

# flights = df.pivot("time", "n_nodes", "influence")
# g1 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax1,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto',center=0)
# g1.set_ylabel('No. Communities')
# g1.set_xlabel('No. Nodes')
# g1.set_title("Original Graph")

# flights = df2.pivot("time", "n_nodes", "influence")
# g2 = sns.heatmap(flights,cmap="YlGnBu",cbar=False,ax=ax2,linewidths=.5, square=False,vmin=1, vmax=max(z),yticklabels='auto')
# g2.set_ylabel('No. Communities')
# g2.set_xlabel('No. Nodes')
# g2.set_title("50%")


# df_fake = pd.DataFrame()
# df_fake["n_nodes"] = [round(x/2,1) for x in df["n_nodes"].to_list()]
# df_fake["influence"] = [x/2 for x in df["influence"].to_list()]
# df_fake["time"] = [x for x in df["time"].to_list()]

# flights = df_fake.pivot("time", "n_nodes", "influence")

# g3 = sns.heatmap(flights,cmap="YlGnBu",ax=ax3, cbar_ax=axcb,linewidths=.5, square=False, vmin=1, vmax=max(z),yticklabels='auto')
# g3.set_ylabel('No. Communities')
# g3.set_xlabel('No. No')
# g3.set_title("Ideal Influence%")


# plt.tight_layout()

# plt.show()