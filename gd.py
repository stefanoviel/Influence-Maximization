from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np

df = pd.read_csv("facebook/facebook_combined-k100-p0.05-IC-degree.csv",sep=",")

df1 = pd.read_csv("facebook/facebook_combined_False-2-k50-p0.05-IC-degree.csv",sep=",")




nodes1 = df["n_nodes"]
comm1 = df["communities"]
inf1 = df["influence"]




nodes2 = df1["n_nodes"]
comm2 = df1["communities"]
inf2 = df1["influence"]



pf = []
A = []

for i in range(len(nodes1)):
    solution = []
    solution.append(inf1[i])
    solution.append(nodes1[i])
    solution.append(comm1[i])
    pf.append(solution)

for i in range(len(nodes2)):
    solution = []
    solution.append(inf2[i])
    solution.append(nodes2[i])
    solution.append(comm2[i])
    A.append(solution)


pf = np.array(pf)
A = np.array(A)

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd import  GD

metric = GD(pf, zero_to_one=True)

igd = metric.do(A)

print(igd)

#gd = get_performance_indicator("igd+", pf)
#print("GD", gd.do(A))