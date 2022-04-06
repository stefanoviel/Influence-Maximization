import pandas as pd


while (True):
    try:
        df = pd.read_csv('soc-brightkite_WC_CELF_runtime.csv',sep=',')
        print(df)
        del df
    except:
        pass