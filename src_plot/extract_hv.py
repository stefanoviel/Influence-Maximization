import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_file_hv(directory): 
    all_hv = []
    all_file = []
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')
            df = df.drop(['generation'], axis=1)
            all_hv.append(df[df['hv_influence_seed'].max() == df['hv_influence_seed']].iloc[-1].to_list())
            all_file.append(file)
    
    all_file = [f.replace('_hv_', '') for f in all_file]
    
    df = [pd.read_csv(os.path.join(directory, f), sep = ',') for f in all_file]
    
    all_hv = np.array(all_hv)
    # print("all_hv")
    # print(all_hv)
    avg_hv = np.mean(all_hv, axis= 0)
    std_hv = np.std(all_hv, axis = 0)
    
    return df, avg_hv, std_hv

def save_hv(hv_influence_seedSize, hv_influence_seedSize_communities, hv_influence_seedSize_time, hv_influence_seedSize_time_communities, file): 

    res = []
    hv_influence_seedSize = hv_influence_seedSize.tolist()
    hv_influence_seedSize.insert(0, "infl_seed")
    res.append(hv_influence_seedSize) 

    hv_influence_seedSize_communities = hv_influence_seedSize_communities.tolist()
    hv_influence_seedSize_communities.insert(0, "infl_seed_com")
    res.append(hv_influence_seedSize_communities) 

    hv_influence_seedSize_time = hv_influence_seedSize_time.tolist()
    hv_influence_seedSize_time.insert(3, None)
    hv_influence_seedSize_time.insert(3, None)
    hv_influence_seedSize_time.insert(0, "infl_seed_time")
    res.append(hv_influence_seedSize_time) 

    hv_influence_seedSize_time_communities = hv_influence_seedSize_time_communities.tolist()
    hv_influence_seedSize_time_communities.insert(0, "infl_seed_com_time")
    res.append(hv_influence_seedSize_time_communities) 

    df = pd.DataFrame(res)
    print(df)

    df.columns = ["fitness_function", "hv_influence_seed", "hv_influence_seedSize_time", "hv_influence_seedSize_communities","hv_influence_communities", "hv_seed_communities", 
                 "hv_influence_time", "hv_seed_time", "hv_influence_seedSize_communities_time"]
    df.to_csv(os.path.join(new_dir, file), float_format='%g')


if __name__ == "__main__":  
    directory = "exp1_out_facebook_combined_4-WC"
    new_dir = os.path.join('result_comparison', directory.replace('exp1_out_', ''))
    try:
        os.makedirs(new_dir)
    except FileExistsError: 
        print("Directory already created")

    show = False 
    fitness_function = "influence_seedSize"
    df_influence_seedSize, avg_hv_influence_seedSize, std_hv_influence_seedSize = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize) 
    print(std_hv_influence_seedSize)

    fitness_function = "influence_seedSize_communities"
    df_influence_seedSize_communities, avg_hv_influence_seedSize_communities, std_hv_influence_seedSize_communities = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_communities) 
    print(std_hv_influence_seedSize_communities)

    fitness_function = "influence_seedSize_time"
    df_influence_seedSize_time, avg_hv_influence_seedSize_time, std_hv_influence_seedSize_time = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_time) 
    print(std_hv_influence_seedSize_time)

    fitness_function = "influence_seedSize_communities_time"
    df_influence_seedSize_time_communities, avg_hv_influence_seedSize_time_communities, std_hv_influence_seedSize_time_communities = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_time_communities) 
    print(std_hv_influence_seedSize_time_communities)

    save_hv(avg_hv_influence_seedSize, avg_hv_influence_seedSize_communities, avg_hv_influence_seedSize_time, avg_hv_influence_seedSize_time_communities, "avg_hv.csv")
    save_hv(std_hv_influence_seedSize, std_hv_influence_seedSize_communities, std_hv_influence_seedSize_time, std_hv_influence_seedSize_time_communities, "std_hv.csv")