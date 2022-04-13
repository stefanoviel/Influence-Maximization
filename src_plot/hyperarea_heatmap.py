import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

if __name__ == '__main__':
   
    name_graph = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
    alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
   
    fig,axn = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(12,7))
    cbar_ax = fig.add_axes([.91, .3, .02, .4])
    ax = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]

    degree_measure = ['degree_centrality','closeness', 'betweenness', 'eigenvector_centrality', 'katz_centrality','page_rank','core']

    for idk, graph in enumerate(name_graph):
        hr = {}
        hr[' '] = [None for x in range(6)]
        hr['MOEA'] = []
        hr[''] = [None for x in range(6)]
        for item in degree_measure:
            hr[item] = []
        model = ['IC', 'WC']
        for m in model:
            file = graph + '-' + m
            df = pd.read_csv('src_plot/moea_results/'+file)
            x = df['Hyperarea']
            x = x[::-1]
            for value in x:
                hr['MOEA'].append(value)
        filenames = ['experiments_upscaling/pfs_upscaling/' + graph +'_IC_2_MAPPING.csv', 'experiments_upscaling/pfs_upscaling/' + graph +'_IC_4_MAPPING.csv', 'experiments_upscaling/pfs_upscaling/' + graph +'_IC_8_MAPPING.csv','experiments_upscaling/pfs_upscaling/' + graph +'_WC_2_MAPPING.csv','experiments_upscaling/pfs_upscaling/' + graph +'_WC_4_MAPPING.csv','experiments_upscaling/pfs_upscaling/' + graph +'_WC_8_MAPPING.csv']
        for item in filenames:
            df = pd.read_csv(item, sep=",")
            measure = df["measure"].to_list()
            hv = df['Hyperarea'].to_list()
            for idx,m in enumerate(measure):
                if m in degree_measure:
                    hr[m].append(hv[idx])

        conf_arr = []
        for key, value in hr.items():
            conf_arr.append(hr[key])
        
        hr_empy= {}
        for key, value in hr.items():
            appo = key
            key = key.replace('_centrality' , '')
            key = key.replace('_' , ' ')
            hr_empy[key] = value
            
        t = ax[idk]
        df = pd.DataFrame.from_dict(hr_empy, orient='index')
        if idk == 5:
            sns.heatmap(df, annot=True,cmap ="YlGnBu",vmin=0, vmax=1, linecolor='white', linewidths=.1, ax= axn[t[0]][t[1]],cbar=True, cbar_ax=cbar_ax,annot_kws={"fontsize":12})
        else:
            sns.heatmap(df, annot=True,cmap ="YlGnBu",vmin=0, vmax=1, linecolor='white', linewidths=.1, ax= axn[t[0]][t[1]],cbar=False, annot_kws={"fontsize":12})
        
        df.iloc[0] = [0,0,0,0,0,0]
        df = df.round(2)
        column_max = df.idxmax(axis=0)
        axn[t[0]][t[1]].set_title(alias[idk], fontsize=14)
        labels = ['IC $\it{s}$=2','IC $\it{s}$=4','IC $\it{s}$=8','WC $\it{s}$=2','WC $\it{s}$=4','WC $\it{s}$=8']
        axn[t[0]][t[1]].set_xticklabels(labels,rotation=90,fontsize=14)
        axn[t[0]][t[1]].set_yticklabels(list(hr_empy.keys()),rotation=0,fontsize=14)
        axn[t[0]][t[1]].tick_params(left=False, bottom=False)
        axn[t[0]][t[1]].text(x=2.1, y=0.5, s='downscaling',fontsize=12)
        axn[t[0]][t[1]].text(x=2.3, y=2.6,s='upscaling',fontsize=12)

    plt.subplots_adjust(left=0.07,
                bottom=0.1, 
                right=0.99, 
                top=0.95, 
                wspace=0.1, 
                hspace=0.1)
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig('Figure-5.eps', format='eps')
    plt.savefig('Figure-5.pdf', format='pdf')
    plt.show()
        
