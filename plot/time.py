import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





df8 = pd.read_csv("fb_politician_TRUE-8.0-k17-p0.2-IC-NEW_3_OBJ-time.csv", sep=",")

p8 = {}
p8["avg"] = np.array([])
p8["std"] = np.array([])
p8["generation"] = np.array([])
p8_total = 0
for i in range(len(df8)):
    t = list(df8.iloc[i])
    p8_total += sum(t)
    p8["avg"] = np.append(p8["avg"], np.mean(t))
    p8["std"] = np.append(p8["std"], np.std(t))
    p8["generation"]  = np.append(p8["generation"],i)


df4 = pd.read_csv("fb_politician_TRUE-4.0-k35-p0.2-IC-NEW_3_OBJ-time.csv", sep=",")

p4 = {}
p4["avg"] = np.array([])
p4["std"] = np.array([])
p4["generation"] = np.array([])
p4_total = 0

for i in range(len(df4)):
    t = list(df4.iloc[i])
    p4_total += sum(t)
    p4["avg"] = np.append(p4["avg"], np.mean(t))
    p4["std"] = np.append(p4["std"], np.std(t))
    p4["generation"]  = np.append(p4["generation"],i)



df2 = pd.read_csv("fb_politician_TRUE-2.0-k70-p0.2-IC-NEW_3_OBJ-time.csv", sep=",")
p2 = {}
p2["avg"] = np.array([])
p2["std"] = np.array([])
p2["generation"] = np.array([])
p2_total = 0

for i in range(len(df2)):
    t = list(df2.iloc[i])
    p2_total += sum(t)
    p2["avg"] = np.append(p2["avg"], np.mean(t))
    p2["std"] = np.append(p2["std"], np.std(t))
    p2["generation"]  = np.append(p2["generation"],i)




df0 = pd.read_csv("fb_politician-k147-p0.2-IC-NEW_3_OBJ-time.csv",sep=',')
p_original = {}
p_original["avg"] = np.array([])
p_original["std"] = np.array([])
p_original["generation"] = np.array([])
p_original_total = 0

for i in range(len(df0)):
    t = list(df0.iloc[i])
    p_original_total += sum(t)
    p_original["avg"] = np.append(p_original["avg"], np.mean(t))
    p_original["std"] = np.append(p_original["std"], np.std(t))
    p_original["generation"] = np.append(p_original["generation"],i)

gen =(p8["generation"])

plt.plot(gen, p8["avg"], label = 'Scale 1/8', color="green")
plt.fill_between(gen, p8["avg"]-p8["std"], p8["avg"]+p8["std"],alpha=0.3, facecolor='green')


plt.plot(gen, p4["avg"], label = 'Scale 1/4', color="orange")
plt.fill_between(gen, p4["avg"]-p4["std"], p4["avg"]+p4["std"],alpha=0.3, facecolor='orange')


plt.plot(gen, p2["avg"], label = 'Scale 1/2', color="blue")
plt.fill_between(gen, p2["avg"]-p2["std"], p2["avg"]+p2["std"],alpha=0.3, facecolor='blue')

plt.plot(gen, p_original["avg"], label = 'Original', color="red")
plt.fill_between(gen, p_original["avg"]-p_original["std"], p_original["avg"]+p_original["std"],alpha=0.3, facecolor='red')


plt.legend()
plt.show()
plt.cla()

objects = ['Scale 1/8', 'Scale 1/4','Scale 1/2', 'Original']
y_pos = np.arange(len(objects))
performance = [p8_total, p4_total, p2_total, p_original_total]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('')

plt.show()