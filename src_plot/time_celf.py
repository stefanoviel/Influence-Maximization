import pandas as pd
import matplotlib.pyplot as plt

graphs = ['soc-gemsec','soc-brightkite']
fig,(ax1, ax2) = plt.subplots(1,2, figsize=(10,2.5), sharey=False, sharex=False)

for index, name in enumerate(graphs):
    if index == 0:
        df16 =  pd.read_csv('experiments_moea/soc-gemsec_16-WC-71/run-1-time.csv')
        SUM16 = df16.values.sum()
        df32 = pd.read_csv('experiments_moea/soc-gemsec_32-WC-35/run-1-time.csv')
        SUM32 = df32.values.sum()
        df_CELF = pd.read_csv('experiments_heuristic/soc-gemsec_WC_CELF_runtime.csv')
        SUM_CELF = list(df_CELF['time'])[0]
        
        v16 = round((1-SUM16/SUM_CELF)*100,1)
        v32 = round((1-SUM_CELF/SUM_CELF)*100,1)
        values = [SUM_CELF, SUM16, SUM32]
        G = ['CELF', ' $\it{s}$=16', ' $\it{s}$=32']
        ax1.bar(G,values,color=['olive', 'brown', 'brown'])
        bars = ax1.patches
        hatch= ['o', '*','.']
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch[i])
        ax1.text(0.80,SUM16+0.02*SUM16, '-{0}%'.format(v16), fontsize=12)
        ax1.text(1.8,SUM32+0.04*SUM32, '-{0}%'.format(v32), fontsize=12)
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 11)
        ax1.set_title(f'{name}', x=0.8, y=0.8,fontsize=12,weight="bold")
        ax1.set_ylabel('Activation Attempts', fontsize=12)
    else:

        df16 =  pd.read_csv('experiments_moea/soc-brightkite_16-WC-66/run-1-time.csv')
        SUM16 = df16.values.sum()
        df32 = pd.read_csv('experiments_moea/soc-brightkite_32-WC-32/run-1-time.csv')
        SUM32 = df32.values.sum()
        df_CELF = pd.read_csv('experiments_heuristic/soc-brightkite_WC_CELF_runtime.csv')
        SUM_CELF = list(df_CELF['time'])[0]
        values = [SUM_CELF, SUM16, SUM32]
        v16 = round((1-SUM16/SUM_CELF)*100,1)
        v32 = round((1-SUM32/SUM_CELF)*100,1)
        values = [SUM_CELF, SUM16, SUM32]
        G = ['CELF', ' $\it{s}$=16', ' $\it{s}$=32']
        ax2.bar(G,values,color=['olive', 'brown', 'brown'])
        hatch= ['o', '*','.']
        bars = ax2.patches
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch[i])
        ax2.text(0.80,SUM16+0.05*SUM16, '-{0}%'.format(v16), fontsize=12)
        ax2.text(1.8,SUM32+0.09*SUM32, '-{0}%'.format(v32), fontsize=12)
        ax2.tick_params(axis = 'both', which = 'major', labelsize = 11)
        ax2.set_title(f'{name}', x=0.8, y=0.8,fontsize=12,weight="bold")

plt.subplots_adjust(left=0.05,
bottom=0.09, 
right=0.99, 
top=0.92, 
wspace=0.12, 
hspace=0.0)
plt.savefig('Figure-7.eps', format='eps')
plt.show()
