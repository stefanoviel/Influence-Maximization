import pandas as pd
import os

filename = "/Users/elia/Downloads/soc-brightkite/soc-brightkite.csv"
name = (os.path.basename(filename))
name = name.replace(".edges","")

df = pd.read_csv(filename, sep=" ",index_col=False)
print(df)
print(df.index)
n1 = df["node1"].to_list()
n2 = df["node2"].to_list()

print(len(set(n1+n2)))

text = []
for i in range(len(n1)):

    f = "{0} {1}".format(n1[i],n2[i])
    text.append(f) 
with open("graphs/soc-brightkite.txt", "w") as outfile:
        outfile.write("\n".join(text))
        