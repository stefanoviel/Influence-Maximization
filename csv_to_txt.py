import pandas as pd
import os

filename = "/Users/elia/Downloads/facebook_large/musae_facebook_edges.csv"
name = (os.path.basename(filename))

df = pd.read_csv(filename)
print(df)

n1 = df["id_1"].to_list()
n2 = df["id_2"].to_list()

text = []
for i in range(len(n1)):

    f = "{0} {1}".format(n1[i],n2[i])
    text.append(f) 
with open("graphs/"+ str(name)+".txt", "w") as outfile:
        outfile.write("\n".join(text))
        