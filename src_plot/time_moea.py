import os 
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_time(path, name):
    df_new = pd.DataFrame()
    scale = ['$\it{s}$=8', '$\it{s}$=4', '$\it{s}$=2', 'Original']
    scale = scale[::-1]
    path = path[::-1]
    for idx, item in enumerate(path):
        activations = []
        for filename in glob.glob(os.path.join(item, '*.csv')):
            if 'time' in filename:
                df = pd.read_csv(filename,sep=',')
                s = 0
                for i in range(len(df)):
                    t = list(df.iloc[i])
                    s += sum(t)
                activations.append(s)
        if idx == 0:
            df_new['Activation Attempts'] =  activations
            df_new['Dataset'] = [name for i in range(len(activations))]
            df_new["Graph"] = [scale[idx] for i in range(len(activations))]

        else:
            df1 = pd.DataFrame()
            df1['Activation Attempts'] =  activations
            df1['Dataset'] = [name for i in range(len(activations))]
            df1["Graph"] = [scale[idx] for i in range(len(activations))]

            df_new = pd.concat([df_new, df1], join="inner")    
    return df_new


if __name__ == '__main__':
    graphs = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure','deezerEU','pgp']
    alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.','Deezer','PGP']
    
    fig,(ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True,figsize=(10,3.5))
    models = ['IC', 'WC']
    for model in models:
        for idx, name in enumerate(graphs):
            path = ["experiments_moea/{0}_8-{1}".format(name,model),"experiments_moea/{0}_4-{1}".format(name,model),"experiments_moea/{0}_2-{1}".format(name,model),"experiments_moea/{0}-{1}".format(name,model)]
            df_results = plot_time(path, alias[idx])
            scale = [8,4,2,1]
            if idx == 0:
                df_ = df_results
            else:
                df_ = pd.concat([df_, df_results], join="inner")
        if model == 'IC':
            ax1.set_yscale('log')     
            bar = sns.barplot(x='Dataset', y='Activation Attempts', hue='Graph', palette=['red', 'orange', 'blue', 'green'], data = df_, ax=ax1)
            ax1.set_title('{0} Model'.format(model), x=0.5, y=0.9, fontsize=12,weight="bold")
            ax1.get_legend().remove()
            ax1.set(xlabel='',ylabel='Activation Attempts')
            ax1.xaxis.get_label().set_fontsize(14)
            ax1.yaxis.get_label().set_fontsize(14)
        else:
            ax2.set_yscale('log')     
            bar = sns.barplot(x='Dataset', y='Activation Attempts',hue='Graph', palette=['red', 'orange', 'blue', 'green'], data = df_, ax=ax2)
            ax2.legend(fontsize=10)
            ax2.set_title('{0} Model'.format(model), x=0.5, y=0.9, fontsize=12,weight="bold")

            ax2.set(xlabel='',ylabel='')
            ax2.xaxis.get_label().set_fontsize(14)
    plt.subplots_adjust(left=0.07,
                bottom=0.09, 
                right=0.99, 
                top=0.97, 
                wspace=0, 
                hspace=0.35)
    plt.savefig('Figure-6.eps', format='eps')
    #plt.savefig('Figure-6.pdf', format='pdf')
    plt.show() 
