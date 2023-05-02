import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_initials(fitness_name):
    if fitness_name == 'influence_seedSize_time':
        return 'I.S.T.'
    elif fitness_name == 'influence_seedSize_communities':
        return 'I.S.C.'
    elif fitness_name == 'influence_seedSize':
        return 'I.S.'
    elif fitness_name == 'influence_seedSize_communities_time':
        return 'I.S.C.T'


facebook_com = pd.DataFrame()
fb_pol = pd.DataFrame()
pgp = pd.DataFrame()
deezer = pd.DataFrame()
fb_org = pd.DataFrame()

dirs = list(os.listdir('result_comparison'))
dirs.remove('aggregate_prob')

for directory in dirs: 

    prob = directory.split('-')[-1]

    if prob == '0.3': 
        continue

    df_combined = pd.DataFrame()
    for file in os.listdir(os.path.join('result_comparison', directory, 'hvs')):
        # print(os.path.join('result_comparison', directory, 'hvs', file))
        df = pd.read_csv(os.path.join(
            'result_comparison', directory, 'hvs', file))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['fitness_function'] = get_initials(file.replace('_hvs.csv', ''))
      
        df_combined = pd.concat([df_combined, df], axis=0)
        
        # print(directory, file)
        # print(len(df_combined))

    df_new = []
    for index, row in df_combined.iterrows():
        df_new.append([row["fitness_function"], "influence seed",
                      row["hv_influence_seedSize"], prob ])
        df_new.append([row["fitness_function"], "influence seed time",
                      row["hv_influence_seedSize_time"], prob ])
        df_new.append([row["fitness_function"], "influence seed communities",
                      row["hv_influence_seedSize_communities"], prob])
    
    df_new = pd.DataFrame(df_new)    
    df_new.columns = ['fitness_function', 'HV dimensions', 'HV values', 'IC probability']
    custom_dict = {'I.S.': 0, 'I.S.C.': 1, 'I.S.T.': 2, 'I.S.C.T':  3}
    df_new = df_new.sort_values(
        by=['fitness_function', 'HV dimensions'], key=lambda x: x.map(custom_dict))

    if 'facebook_combined' in directory: 
        facebook_com = pd.concat([facebook_com, df_new], axis=0)

    if 'deezerEU' in directory: 
        deezer = pd.concat([deezer, df_new], axis=0)

    if 'fb_politician' in directory: 
        fb_pol = pd.concat([fb_pol, df_new], axis=0)

    if 'pgp' in directory: 
        pgp = pd.concat([pgp, df_new], axis=0)
    
    if 'fb_org' in directory: 
        fb_org = pd.concat([fb_org, df_new], axis=0)


# print(facebook_com.loc[facebook_com.prob == '0.05'])
    
# sns.catplot(data=facebook_com.loc[(facebook_com['HV dimensions'] == 'infl. seed' )& (facebook_com['prob'] == '0.05' )], x='prob', y='HV values', kind="bar", hue="fitness_function",
#             legend=False, height=5, aspect=6/5,  palette= sns.color_palette("deep"))

hbar = False



if hbar: 
    for db, name in  zip([facebook_com, fb_pol, pgp, deezer, fb_org], ['facebook_com',  'fb_pol', 'pgp', 'deezer', 'fb_org']): 

        db = db.sort_values(by=['fitness_function', 'IC probability'], ascending=True)
        

        for dimensions in ['influence seed communities', 'influence seed', 'influence seed time']: 
            print(db.reset_index())
            g = sns.catplot(data=db.loc[db['HV dimensions'] == dimensions], x='IC probability', y='HV values', kind="bar", hue="fitness_function",
                        legend=False, height=5, aspect=6/5,  palette= sns.color_palette("deep"), log = False)
            
            # order = ['I.S.', 'I.S.C.', 'I.S.T.', 'I.S.C.T']
            # g.set(ylabel='log(HV values)')
            plt.legend(loc='upper left', title='objectives')
            # plt.savefig(os.path.join('result_comparison', 'aggregate_prob', name + '_' + dimensions + '.pdf'), format="pdf")
            print(name, dimensions)
            plt.show()

else: 

    for db, name in zip([facebook_com, fb_pol, pgp, deezer, fb_org], ['facebook_com',  'fb_pol', 'pgp', 'deezer', 'fb_org']): 
        db = db.sort_values(by=['IC probability', 'HV dimensions'], ascending=True)
        print(name)
        # db = db.loc[db['HV dimensions'] != 'influence seed']

        df = db.reset_index(drop=True)
        # df['HV values'] = - np.log(df['HV values'])

        # sns.set(rc={'figure.figsize':(8,5)})
        plt.figure(figsize=(8,5))
        g_results = sns.lineplot(data=df, x='IC probability', y="HV values", hue="HV dimensions")
        g_results.set(yscale='log')
        g_results.set( ylabel='log(HV values)')
        plt.savefig(os.path.join('result_comparison', 'aggregate_prob', name + '_' + 'comp_time_com' + '.pdf'), format="pdf")
        plt.show()