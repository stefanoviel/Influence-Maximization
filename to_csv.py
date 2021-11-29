import pandas as pd


df = pd.read_csv("/Users/elia/Desktop/facebook_1912/edges.csv",sep=",")
node1 = df["node_1"]
node2 = df["node_2"]


text = []
for i in range(len(node1)):
    f = "{0} {1}".format(node1[i],node2[i])
    text.append(f)

with open("facebook_1912.txt", "w") as outfile:
        outfile.write("\n".join(text))
