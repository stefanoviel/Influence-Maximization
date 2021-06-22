from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/elia/Desktop/Influence-Maximization/RandomGraph-N200-E5942-population.csv")

#df = df.sort_values(by="generations").reset_index()
#df = df.drop_duplicates(subset=["generations"], keep='last')

print(df)
print(df.columns)
y = df["generations"]
x = df["influence"]

plt.scatter(x, y)
x = list(set(x.sort_values()))
# y = []
# a = []
# for item in x:
#     df1 = df[df.generations == item]
#     #a = df1["generations"].to_list()
    
#     if df1["n_nodes"].max() >

#     y.append(df1["n_nodes"].max())
#     a.append(df1["n_nodes"].max())


#print(x)
#print(y)
#plt.plot(x,y,marker='o', color="red")
plt.xlabel('Fitness')
plt.ylabel('Generations')

plt.show()