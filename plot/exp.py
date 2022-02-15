import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np
import glob



def plot_images(path, color):
    t = []
    for filename in glob.glob(os.path.join(path, '*.csv')):
        if '_hv' in filename:
            df = pd.read_csv(filename, sep=',')
            t.append(list(df["hv"]))

    m = []
    s = []
    for i in range(len(t[0])):
        k = []
        for j in range(len(t)):
            k.append(t[j][i])
        m.append(np.mean(k))
        s.append(np.std(k))
    m = np.array(m)
    s = np.array(s)
    plt.plot(list(df["generation"]), m,color=color)
    plt.fill_between(list(df["generation"]), m-s, m+s, alpha=0.3, color=color)
    #plt.errorbar(list(df["generation"]), m,s, linestyle='None', marker='^')



def plot_time(path):
    m = []
    std = []
    for item in path:
        tot = []
        for filename in glob.glob(os.path.join(item, '*.csv')):
            if 'time' in filename:

                df = pd.read_csv(filename,sep=',')
                s = 0
                for i in range(len(df)):
                    t = list(df.iloc[i])
                    s += sum(t)
                tot.append(s)
        m.append(np.mean(tot))
        std.append(np.std(tot))

    objects = ['Scale 1/8', 'Scale 1/4','Scale 1/2', 'Original']
    y_pos = np.arange(len(objects))
    #performance = [p8_total, p4_total, p2_total, p_original_total]
    performance = [m[0], m[1],m[2],m[3]]
    #plt.yscale('log')
    plt.bar(y_pos, performance, yerr=[std[0], std[1], std[2], std[3]],align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('Cumulative Time')
    plt.xlabel('Graphs')
    #plt.title('Time Fb PolitLTians WC MODEL')
    #plt.ylim((0,1000))
    plt.savefig('pgp_time_LT')

path = 'experiments/pgp_8-LT'
plot_images(path, 'green')


path ='experiments/pgp_4-LT'
plot_images(path, 'blue')

path ='experiments/pgp_2-LT'
plot_images(path, 'orange')

path ='experiments/pgp-LT'
plot_images(path, 'red')


plt.savefig('pgp_hv_LT')




plt.cla()
plt.close()

path = ["experiments/pgp_8-LT","experiments/pgp_4-LT","experiments/pgp_2-LT","experiments/pgp-LT"]

plot_time(path)



exit(0)


