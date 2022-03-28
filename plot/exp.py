from tkinter import FALSE
import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np
import glob
import seaborn as sns

#sns.set(font_scale = 2)
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

def plot_time(path, name):
    m = []
    std = []
    df_new = pd.DataFrame()
    scale = ['$\it{s}$=8', '$\it{s}$=4', '$\it{s}$=2', 'Original']
    scale = scale[::-1]
    path = path[::-1]
    for idx, item in enumerate(path):
        tot = []
        for filename in glob.glob(os.path.join(item, '*.csv')):
            if 'time' in filename:

                df = pd.read_csv(filename,sep=',')
                s = 0
                for i in range(len(df)):
                    t = list(df.iloc[i])
                    s += sum(t)
                tot.append(s)
        if idx == 0:
            df_new['Activation Attempts'] =  tot
            df_new['Dataset'] = [name for i in range(len(tot))]
            df_new["Graph"] = [scale[idx] for i in range(len(tot))]

        else:
            df1 = pd.DataFrame()
            df1['Activation Attempts'] =  tot
            df1['Dataset'] = [name for i in range(len(tot))]
            df1["Graph"] = [scale[idx] for i in range(len(tot))]

            df_new = pd.concat([df_new, df1], join="inner")

    #plt.bar(y_pos, performance, yerr=[std[0], std[1], std[2], std[3]],align='center', alpha=0.5, ecolor='black', capsize=10)
    #plt.xticks(y_pos, objects)
    #plt.ylabel('Cumulative Time')
    #plt.xlabel('Graphs')
    #plt.plot(objects, performance)
 
    #plt.errorbar(objects, performance , yerr=[std[0], std[1], std[2], std[3]], fmt ='o')    
    #plt.savefig('pgp_time_LT')
    #plt.show()
    
    return df_new


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
graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure','deezerEU','pgp']
alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.','Deezer','PGP']
fig,(ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True,figsize=(10,3.5))
models = ['IC', 'WC']
for model in models:
    TIME = {}
    for g in graphs:
        TIME[g] = []
    df_ = pd.DataFrame()
    for idx, name in enumerate(graphs):
        '''
        path8 = 'experiments/{0}_8-{1}'.format(name, model)
        m8 = plot_images(path8, 'green')


        path4 ='experiments/{0}_4-{1}'.format(name, model)
        m4= plot_images(path4, 'blue')

        path2 ='experiments/{0}_2-{1}'.format(name, model)
        m2 = plot_images(path2, 'orange')

        path ='experiments/{0}-{1}'.format(name, model)
        m1= plot_images(path, 'red')



        gd8 = gen_dist(path, path8)
        gd4 = gen_dist(path, path4)
        gd2 = gen_dist(path, path2)   
        gd1 = gen_dist(path, path)   


        print(gd8, gd4,gd2)
        df = pd.DataFrame()
        df["scale"] = [8,4,2]
        df["Hyperarea"] = [m8/m1,m4/m1,m2/m1]
        df["GD"] = [gd8,gd4,gd2]
        path = path.replace('experiments/','')
        df.to_csv('matrix_MOEA_results/'+path, sep=',', index=False)
        '''


        path = ["experiments/{0}_8-{1}".format(name,model),"experiments/{0}_4-{1}".format(name,model),"experiments/{0}_2-{1}".format(name,model),"experiments/{0}-{1}".format(name,model)]
        print(name)
        df_results = plot_time(path, alias[idx])
        scale = [8,4,2,1]
        if idx == 0:
            df_ = df_results
        else:
            df_ = pd.concat([df_, df_results], join="inner")
    

    if model == 'IC':
        ax1.set_yscale('log')     
        bar = sns.barplot(x='Dataset', y='Activation Attempts', hue='Graph', palette=['red', 'orange', 'blue', 'green'], data = df_, ax=ax1)
        #ax1.set_title('{0} Model'.format(model), x=0.5, y=0.9, fontsize=14)
        ax1.get_legend().remove()
        ax1.set(xlabel='',ylabel='Activation Attempts')
        ax1.xaxis.get_label().set_fontsize(14)
        ax1.yaxis.get_label().set_fontsize(14)

    else:
        ax2.set_yscale('log')     

        bar = sns.barplot(x='Dataset', y='Activation Attempts',hue='Graph', palette=['red', 'orange', 'blue', 'green'], data = df_, ax=ax2)
        ax2.legend(fontsize=10)
        #ax2.set_title('{0} Model'.format(model), x=0.5, y=0.9, fontsize=14)
        ax2.set(xlabel='',ylabel='')
        ax2.xaxis.get_label().set_fontsize(14)
    # import itertools
    # hatches = itertools.cycle(['/', '-', 'x','*', 'o', '.'])
    # for i, bar in enumerate(bar.patches):
    #     if i % 4 == 0:
    #         hatch = next(hatches)
    #     bar.set_hatch(hatch)
    df = pd.DataFrame.from_dict(TIME, orient='index')
    df.to_latex('A',index=True)
    print(df)

plt.subplots_adjust(left=0.07,
            bottom=0.09, 
            right=0.99, 
            top=0.97, 
            wspace=0, 
            hspace=0.35)
#fig.legend(loc = 'upper center', ncol=3,
#        bbox_transform = plt.gcf().transFigure)

plt.savefig('time_log.eps', format='eps')
plt.savefig('time_log-eps-converted-to.pdf', format='pdf')
plt.show() 
