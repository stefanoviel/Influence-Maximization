import pandas as pd
import os

filename = "/Users/elia/Downloads/ego-gplus/ego-gplus.edges"
name = (os.path.basename(filename))

df = pd.read_csv(filename, sep=",")
print(df)
print(df.index)
n1 = df["node_1"].to_list()
n2 = df["node_2"].to_list()

text = []
for i in range(len(n1)):

    f = "{0} {1}".format(n1[i],n2[i])
    text.append(f) 
with open("graphs/"+ str(name)+".txt", "w") as outfile:
        outfile.write("\n".join(text))
        