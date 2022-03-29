import pandas as pd
import numpy as np


name_graph = ['facebook_combined',  'fb_politician', 'fb_org', 'fb-pages-public-figure', 'pgp','deezerEU']
alias = ['Ego Fb.','Fb. Pol.', 'Fb. Org.', 'Fb. Pag.', 'PGP','Deezer']
RESULTS = {}
TEST = {}
model = 'IC'
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
        p_values.append(format(res.pvalue, '.3g'))
        res = ttest_ind(m4, m1)
        p_values.append(format(res.pvalue, '.3g'))
        res = ttest_ind(m2, m1)
        p_values.append(format(res.pvalue, '.3g'))
        res = ttest_ind(m1, m1)
        print(p_values)
        p_values = p_values[::-1]
    except:
        pass

    RES[alias[idx]] = p_values

df = pd.DataFrame.from_dict(RES,orient='index')

df.to_latex('p_value_after_mapping_{0}'.format(model))


