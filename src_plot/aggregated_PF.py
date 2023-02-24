# compute avg and std tables which can be found in result_comparison.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_file_hv(directory): 
    """
    From each file extract only all the hv corresponding to the generation that achieved the best results on hv_influence_seed
    compute the average and std on those values among all the hv elements
    """
    all_hv = []
    all_file = []
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')        
            # get best hv value of the run
            all_hv.append(df[df['hv_influence_seed'].max() == df['hv_influence_seed']].iloc[-1].to_list())
            all_file.append(file)
    
    all_file = [f.replace('_hv_', '') for f in all_file]

    df = [pd.read_csv(os.path.join(directory, f), sep = ',') for f in all_file]
    all_hv = np.array(all_hv)  # TODO: save all_hv, so that you can do the scatter plot on them without matplotlib
    avg_hv = np.mean(all_hv, axis= 0)[1:]
    std_hv = np.std(all_hv, axis = 0)[1:]

    return df, avg_hv, std_hv

def save_hv(hv_influence_seedSize, hv_influence_seedSize_communities, hv_influence_seedSize_time, hv_influence_seedSize_time_communities, file): 
    """
    Given hv values get saved in a file
    """
    res = []
    hv_influence_seedSize = hv_influence_seedSize.tolist()
    hv_influence_seedSize.insert(0, "infl_seed_com")
    res.append(hv_influence_seedSize) 

    hv_influence_seedSize_communities = hv_influence_seedSize_communities.tolist()
    hv_influence_seedSize_communities.insert(0, "infl_seed_com")
    res.append(hv_influence_seedSize_communities) 

    hv_influence_seedSize_time = hv_influence_seedSize_time.tolist()
    hv_influence_seedSize_time.insert(1, None)
    hv_influence_seedSize_time.insert(1, None)
    hv_influence_seedSize_time.insert(1, None)
    hv_influence_seedSize_time.insert(0, "infl_seed_time")
    res.append(hv_influence_seedSize_time) 

    hv_influence_seedSize_time_communities = hv_influence_seedSize_time_communities.tolist()
    hv_influence_seedSize_time_communities.insert(0, "infl_seed_com_time")
    res.append(hv_influence_seedSize_time_communities) 

    df = pd.DataFrame(res)
    print(df)

    df.columns = ["fitness_function", "hv_influence_seed", "hv_influence_communities","hv_seed_communities", "hv_influence_seedSize_communities", 
                "hv_influence_time", "hv_seed_time", "hv_influence_seedSize_time", "hv_influence_seedSize_communities_time"]
    df.to_csv(os.path.join(new_dir, file), float_format='%g')


if __name__ == "__main__":  

    directory = "exp1_out_facebook_combined_4-IC"
    
    show = True
    fitness_function = "influence_seedSize"
    df_influence_seedSize, avg_hv_influence_seedSize, std_hv_influence_seedSize = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize) 
    print(std_hv_influence_seedSize)

    for df in df_influence_seedSize: 
        plt.scatter(df["n_nodes"], df["influence"], color = "b")
    
    plt.xlabel('% Influenced Nodes',fontsize=12)
    plt.ylabel('% Nodes as seed set',fontsize=12)
    plt.legend()
    new_dir = os.path.join('result_comparison', directory.replace('exp1_out_', ''))
    try:
        os.makedirs(new_dir)
    except FileExistsError: 
        print("Directory already created")
    
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()

    fitness_function = "influence_seedSize_communities"
    df_influence_seedSize_communities, avg_hv_influence_seedSize_communities, std_hv_influence_seedSize_communities = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_communities) 
    print(std_hv_influence_seedSize_communities)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    for df in df_influence_seedSize_communities: 
        axs[0, 0].scatter(df["n_nodes"], df["influence"], color = "b")
        axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
        axs[0, 1].scatter(df["n_nodes"], df["communities"], color = "b")
        axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 1].set_ylabel('Number of Communities',fontsize=12)
        axs[1, 0].scatter(df["influence"], df["communities"], color = "b")
        axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
        axs[1, 0].set_ylabel('Number of Communities',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        print('saving in', os.path.join(new_dir, fitness_function + ".png"))
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    fitness_function = "influence_seedSize_time"
    df_influence_seedSize_time, avg_hv_influence_seedSize_time, std_hv_influence_seedSize_time = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_time) 
    print(std_hv_influence_seedSize_time)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    for df in df_influence_seedSize_time: 
        axs[0, 0].scatter(df["n_nodes"], df["influence"], color = "b")
        axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
        axs[0, 1].scatter(df["n_nodes"], np.log10(df["time"]), color = "b")
        axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 1].set_ylabel('log(1/Time)',fontsize=12)
        axs[1, 0].scatter(df["influence"], df["time"], color = "b")
        axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
        axs[1, 0].set_ylabel('1/Time',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    fitness_function = "influence_seedSize_communities_time"
    df_influence_seedSize_time_communities, avg_hv_influence_seedSize_time_communities, std_hv_influence_seedSize_time_communities = get_file_hv(os.path.join(directory, fitness_function))
    print(avg_hv_influence_seedSize_time_communities) 
    print(std_hv_influence_seedSize_time_communities)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    for df in df_influence_seedSize_time_communities: 
        axs[0, 0].scatter(df["n_nodes"], df["influence"], color = "b")
        axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
        axs[0, 1].scatter(df["n_nodes"], df["time"], color = "b")
        axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
        axs[0, 1].set_ylabel('1/Time',fontsize=12)
        axs[1, 0].scatter(df["influence"], df["time"], color = "b")
        axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
        axs[1, 0].set_ylabel('1/Time',fontsize=12)
        axs[1, 1].scatter(df["communities"], df["time"], color = "b")
        axs[1, 1].set_xlabel('Number of Communities',fontsize=12)
        axs[1, 1].set_ylabel('1/Time',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    save_hv(avg_hv_influence_seedSize, avg_hv_influence_seedSize_communities, avg_hv_influence_seedSize_time, avg_hv_influence_seedSize_time_communities, "avg_hv.csv")
    save_hv(std_hv_influence_seedSize, std_hv_influence_seedSize_communities, std_hv_influence_seedSize_time, std_hv_influence_seedSize_time_communities, "std_hv.csv")