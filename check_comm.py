import pandas as pd
import numpy as np


graphs = ['fb_politician','fb-pages-public-figure', 'facebook_combined', 'fb_org','pgp','deezerEU']


for name in graphs:
    df = pd.read_csv(f'comm_ground_truth/{name}.csv',sep=",")
    df1 = df.groupby('comm')['node'].apply(list).reset_index(name='new')
    n_items = [len(x) for x in df1["new"].to_list()]
    print(name, len(np.unique(df["comm"].to_list())), min(n_items), max(n_items))
