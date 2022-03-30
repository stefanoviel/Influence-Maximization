import pandas as pd
import numpy as np


name_graph = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
RESULTS = {}
TEST = {}
model = 'WC'
scale_factors = [2,4,8]
for idx, name in enumerate(name_graph):
    RESULTS[alias[idx]] = []
    res = []
    for s in scale_factors:
        TEST[f'{name}_{s}'] = []
        HV = []
        hr = []
        try:
            for j in range(10):
                for jj in range(10):
                    df = pd.read_csv(f'Mapping_Results_Script/{name}_{model}_{s}_MAPPING_{j+1}-{jj+1}.csv', sep=',')
                    pg = df[df["measure"] == 'page_rank'].Hyperarea.item()
                    hr.append(pg)
                    print(pg)

                    h = df[df["measure"] == 'page_rank'].HyperVolume.item()
                    HV.append(h)
        except:
            pass

        res.append(np.mean(hr))
        TEST[f'{name}_{s}'].append(HV)
    RESULTS[alias[idx]]=res

df = pd.DataFrame.from_dict(RESULTS, orient='index')
print(df)


for idx, name in enumerate(name_graph):
    TEST[f'{name}_1'] = []
    H = []
    try:
        for i in range(10):
            filename_original_results = "experiments/{0}-{1}/run-{2}_hv_.csv".format(name, model,i+1)
            df = pd.read_csv(filename_original_results, sep= ',')
            hv = df[df['generation'] == 1001].hv.item()
            print('.....',hv)
            H.append(hv)
    except:
        pass

    TEST[f'{name}_1']= H

#print(TEST)

RES = {} 
for idx, name in enumerate(name_graph):
    print('Name -->', name)
    try:
        m1 = np.array(TEST[f'{name}_1'])
        m2 = np.array(TEST[f'{name}_2'][0])
        m4 = np.array(TEST[f'{name}_4'][0])
        m8 = np.array(TEST[f'{name}_8'][0])

        from scipy.stats import ttest_ind
        p_values = []
        res = ttest_ind(m8, m1)
        p_values.append((format(res.pvalue, '.3g')))
        res = ttest_ind(m4, m1)
        p_values.append((format(res.pvalue, '.3g')))
        res = ttest_ind(m2, m1)
        p_values.append((format(res.pvalue, '.3g')))
        res = ttest_ind(m1, m1)
        print(p_values)
        p_values = p_values[::-1]
    except:
        pass

    RES[alias[idx]] = p_values

df = pd.DataFrame.from_dict(RES,orient='index')

df.to_latex('p_value_after_mapping_{0}'.format(model))



print(df)

import seaborn as sns
import matplotlib.pyplot as plt

two = []
four = []
eight = []

two = df[0].to_list()
four = df[1].to_list()
eight = df[2].to_list()

print(eight)


for i in range(len(two)):
    if two[i] < 0.05:
        two[i] = u'\u2713'
    else:
        two[i] = u'U+2715'

for i in range(len(four)):
    if four[i] < 0.05:
        four[i] = u'\u2713'
    else:
        four[i] = u'U+2715'
for i in range(len(eight)):
    if eight[i] < 0.05:
        eight[i] = u'\u2713'
    else:
        eight[i] = u'U+2715'

df[0] = two
df[1] = four
df[2] = eight
fig, ax = plt.subplots()

harvest = []
c = []
for i in range(len(df)):
    harvest.append([0,0,0])
    c.append(list(df.iloc[i]))
harvest = np.array(harvest)
im = ax.imshow(harvest, vmin=0, vmax=0,cmap='gray_r')
print(c)

vegetables = alias 
farmers = ['$\it{s}$=2','$\it{s}$=4','$\it{s}$=8']
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, c[i][j],
                       ha="center", va="center", color="black")

ax.set_title("in tons/year)")
fig.tight_layout()
plt.savefig('prova_p.png')
plt.show()
