from ast import Break
import pandas as pd

import matplotlib.pyplot as plt

graphs = ['soc-gemsec','soc-brightkite']
fig,(ax1, ax2) = plt.subplots(1,2, figsize=(10,4.5), sharey=False, sharex=False)

for index, name in enumerate(graphs):
    if index == 0:
        df0 =  pd.read_csv('experiments/soc-gemsec_16-WC-71/run-1-time.csv')

        SUM0 = df0.values.sum()
        df = pd.read_csv('experiments/soc-gemsec_32-WC-35/run-1-time.csv')

        SUM1 = df.values.sum()

        print(round(df.values.sum(),1))


        df1 = pd.read_csv('soc-gemsec_WC_CELF_runtime.csv')
        SUM2 = list(df1['time'])[0]



        print('% Variation', (1-SUM0/SUM2)*100)
        print('% Variation', (1-SUM1/SUM2)*100)

        v16 = round((1-SUM0/SUM2)*100,1)
        v32 = round((1-SUM1/SUM2)*100,1)

        values = [SUM2, SUM0, SUM1]
        G = ['Original', ' $\it{s}$=16', ' $\it{s}$=32']
        ax1.bar(G,values,color=['olive', 'gray', 'gray'])
        bars = ax1.patches
        hatch= ['o', '*','.']
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch[i])

        ax1.text(0.80,SUM0+0.02*SUM0, '-{0}%'.format(v16), fontsize=12)
        ax1.text(1.8,SUM1+0.04*SUM1, '-{0}%'.format(v32), fontsize=12)
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 11)
        ax1.set_title(f'{name}', x=0.8, y=0.9,fontsize=12)
        ax1.set_ylabel('Activation Attempts', fontsize=12)
    else:

        df0 =  pd.read_csv('experiments/soc-brightkite_16-WC-66/run-1-time.csv')

        SUM0 = df0.values.sum()
        df = pd.read_csv('experiments/soc-brightkite_32-WC-32/run-1-time.csv')

        SUM1 = df.values.sum()

        print(round(df.values.sum(),1))


        df1 = pd.read_csv('soc-brightkite_WC_CELF_runtime.csv')
        SUM2 = list(df1['time'])[0]



        print('% Variation', (1-SUM0/SUM2)*100)
        print('% Variation', (1-SUM1/SUM2)*100)

        values = [SUM2, SUM0, SUM1]
        v16 = round((1-SUM0/SUM2)*100,1)
        v32 = round((1-SUM1/SUM2)*100,1)

        values = [SUM2, SUM0, SUM1]
        G = ['Original', ' $\it{s}$=16', ' $\it{s}$=32']
        ax2.bar(G,values,color=['olive', 'grey', 'grey'])
        hatch= ['o', '*','.']
        bars = ax2.patches
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch[i])
        ax2.text(0.80,SUM0+0.05*SUM0, '-{0}%'.format(v16), fontsize=12)
        ax2.text(1.8,SUM1+0.09*SUM1, '-{0}%'.format(v32), fontsize=12)
        ax2.tick_params(axis = 'both', which = 'major', labelsize = 11)
        ax2.set_title(f'{name}', x=0.8, y=0.9,fontsize=12)

plt.tick_params(axis='both', which='minor', labelsize=12)
plt.subplots_adjust(left=0.05,
bottom=0.08, 
right=0.99, 
top=0.95, 
wspace=0.1, 
hspace=0.0)
plt.savefig('prova.png')
plt.savefig('activations.eps', format='eps')
plt.show()
