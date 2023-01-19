import os
import pandas as pd
import matplotlib.pyplot as plt



directory = "exp1_out_facebook_combined_4-IC"
new_dir = os.path.join('result_comparison', directory.replace('exp1_out_', ''))
df = pd.read_csv(os.path.join(new_dir, "avg_hv.csv"))

fig, axs = plt.subplots(4, 2)
fig.set_size_inches(13, 10)
fig.tight_layout(pad=2.5)

axs[0,0].bar(df["fitness_function"], df["hv_influence_seed"])
axs[0,0].set_title("hv_influence_seed")

no_nan = df["hv_influence_communities"].dropna()
labels = df[pd.isnull(df["hv_influence_communities"])]["fitness_function"] 
axs[0,1].bar(labels, no_nan)
axs[0,1].set_title("hv_influence_communities")

no_nan = df["hv_seed_communities"].dropna()
labels = df[pd.isnull(df["hv_seed_communities"])]["fitness_function"] 
axs[1,0].bar(labels, no_nan)
axs[1,0].set_title("hv_seed_communities")

no_nan = df["hv_influence_time"].dropna()
labels = df[pd.isnull(df["hv_influence_time"])]["fitness_function"] 
axs[1,1].bar(labels, no_nan)
axs[1,1].set_title("hv_influence_time")

no_nan = df["hv_seed_time"].dropna()
labels = df[pd.isnull(df["hv_seed_time"])]["fitness_function"] 
axs[2,0].bar(labels, no_nan)
axs[2,0].set_title("hv_seed_time")

no_nan = df["hv_influence_seedSize_time"].dropna()
labels = df[pd.isnull(df["hv_influence_seedSize_time"])]["fitness_function"] 
axs[2,1].bar(labels, no_nan)
axs[2,1].set_title("hv_influence_seedSize_time")

no_nan = df["hv_influence_seedSize_communities"].dropna()   
labels = df[pd.isnull(df["hv_influence_seedSize_communities"])]["fitness_function"] 
axs[3,0].bar(labels, no_nan)
axs[3,0].set_title("hv_influence_seedSize_communities")

plt.savefig(os.path.join(new_dir, "hv_seed_time" + ".png"))
plt.show()

