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
    #plt.ylim(0,1)
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
    plt.plot(objects, performance)
 
    #plt.errorbar(objects, performance , yerr=[std[0], std[1], std[2], std[3]], fmt ='o')    
    #plt.savefig('pgp_time_LT')
    plt.show()


def gen_dist(path_original,path_scale):
    print(path_scale, path_original)
    gd_list = []
    for i in range(10):
        for j in range(10):
            try:
                df = pd.read_csv(path_scale +  '/run-{0}.csv'.format(i+1))
                z = df["influence"].to_list()
                x = df["n_nodes"].to_list()
                df = pd.read_csv(path_original +  '/run-{0}.csv'.format(j+1))
                z1 = df["influence"].to_list()
                x1 = df["n_nodes"].to_list()
            except:
                pass
           
            pf = []
    
            for k in range(len(x1)):
                pf.append([z1[k], x1[k]])
            A = []
    
            for k in range(len(x)):
                A.append([z[k], x[k]])

            
            from pymoo.factory import get_performance_indicator
            pf = np.array(pf)
            A = np.array(A)

            gd = get_performance_indicator("gd", pf)
            gd_list.append(gd.do(A))

            #print(gd.do(A), i+1, j+1)

    return np.mean(gd_list)
graphs = ['pgp', 'fb_politician','fb-pages-public-figure', 'facebook_combined', 'fb_org', 'deezerEU']
models = ['IC', 'WC']
for name in graphs:
    for model in models:
        path8 = 'experiments/{0}_8-{1}'.format(name, model)
        m8 = plot_images(path8, 'green')


        path4 ='experiments/{0}_4-{1}'.format(name, model)
        m4= plot_images(path4, 'blue')

        path2 ='experiments/{0}_2-{1}'.format(name, model)
        m2 = plot_images(path2, 'orange')

        path ='experiments/{0}-{1}'.format(name, model)
        m1= plot_images(path, 'red')



        #pWC.savefig('fb_politWCian_hv_')
        #plt.show()
        #plt.cla()
        #plt.close()

        gd8 = gen_dist(path, path8)
        gd4 = gen_dist(path, path4)
        gd2 = gen_dist(path, path2)   
        #gd1 = gen_dist(path, path)   


        print(gd8, gd4,gd2)
        df = pd.DataFrame()
        df["scale"] = [8,4,2]
        df["Hyperarea"] = [m8/m1,m4/m1,m2/m1]
        df["GD"] = [gd8,gd4,gd2]
        path = path.replace('experiments/','')
        df.to_csv('matrix_MOEA_results/'+path, sep=',', index=False)
        #print(x,y)
        #plt.plot(x,y, 'go-')
        #plt.show()



        path = ["experiments/{0}_8-{1}".format(name,model),"experiments/{0}_4-{1}".format(name,model),"experiments/{0}_2-{1}".format(name,model),"experiments/{0}-{1}".format(name,model)]

        #,modelpath = ["experiments/{0}_8-IC".format(name), "experiments/pgp_4-IC"]
        #plot_time(path)





