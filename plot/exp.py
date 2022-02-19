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
    plt.ylim(0,1)
    #plt.errorbar(list(df["generation"]), m,s, linestyle='None', marker='^')

    return max(m)

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
    #objects = [8,4,2,1]
    y_pos = np.arange(len(objects))
    performance = [m[0], m[1],m[2],m[3]]
    plt.bar(y_pos, performance, yerr=[std[0], std[1], std[2], std[3]],align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('Cumulative Time')
    plt.xlabel('Graphs')
    #plt.plot(objects, performance)
 
    #plt.errorbar(objects, performance , yerr=[std[0], std[1], std[2], std[3]], fmt ='o')    
    #plt.savefig('pgp_time_LT')
    plt.show()
path = 'experiments/facebook_combined_8-WC'
m8 = plot_images(path, 'green')


path ='experiments/facebook_combined_4-WC'
m4= plot_images(path, 'blue')

path ='experiments/facebook_combined_2-WC'
m2 = plot_images(path, 'orange')

path ='experiments/facebook_combined-WC'
m1= plot_images(path, 'red')



#pIC.savefig('fb_politician_hv_')
plt.show()
plt.cla()
plt.close()



x = [8,4,2,1]
y = [m8, m4, m2,m1]
df = pd.DataFrame()
df["scale"] = [8,4,2]
df["HyperArea"] = [m8/m1,m4/m1,m2/m1]
name = path.replace('experiments/','')
df.to_csv('matrix_MOEA_results/'+name, sep=',', index=False)
#print(x,y)
#plt.plot(x,y, 'go-')
#plt.show()


exit(0)

path = ["experiments/fb_politician_8-IC","experiments/fb_politician_4-IC","experiments/fb_politician_2-IC","experiments/fb_politician-IC"]

#path = ["experiments/pgp_8-IC", "experiments/pgp_4-IC"]
plot_time(path)



exit(0)


