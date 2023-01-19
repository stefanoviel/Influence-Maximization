import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_best(directory): 
    best_hv = 0
    best_file_hv = ''
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')
            if best_hv < df['hv_influence_seed'].max(): 
                best_hv = df['hv_influence_seed'].max()
                best_file_hv = file
    
    best_file = best_file_hv.replace('_hv_', '')
    return best_file, best_file_hv


if __name__ == "__main__":  

    directory = "exp1_out_facebook_combined_4_on_seed-WC"
    
    show = True
    fitness_function = "influence_seedSize"
    best_file, best_file_hv = find_best(os.path.join(directory, fitness_function))
    df_influence_seedSize_hv = pd.read_csv(os.path.join(directory, fitness_function, best_file_hv), sep = ',')
    df_influence_seedSize = pd.read_csv(os.path.join(directory, fitness_function, best_file), sep = ',')
   
    max_hv_influence_seedSize = df_influence_seedSize_hv.iloc[-1]
    print(max_hv_influence_seedSize)

    plt.scatter(df_influence_seedSize["n_nodes"], df_influence_seedSize["influence"])
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
    best_file, best_file_hv = find_best(os.path.join(directory, fitness_function))
    df_influence_seedSize_communities_hv = pd.read_csv(os.path.join(directory, fitness_function, best_file_hv), sep = ',')
    df_influence_seedSize_communities = pd.read_csv(os.path.join(directory, fitness_function, best_file), sep = ',')
    
    max_hv_influence_seedSize_communities = df_influence_seedSize_communities_hv.iloc[-1]
    print(max_hv_influence_seedSize_communities)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    axs[0, 0].scatter(df_influence_seedSize_communities["n_nodes"], df_influence_seedSize_communities["influence"])
    axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
    axs[0, 1].scatter(df_influence_seedSize_communities["n_nodes"], df_influence_seedSize_communities["communities"])
    axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 1].set_ylabel('Number of Communities',fontsize=12)
    axs[1, 0].scatter(df_influence_seedSize_communities["influence"], df_influence_seedSize_communities["communities"])
    axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
    axs[1, 0].set_ylabel('Number of Communities',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    fitness_function = "influence_seedSize_time"
    best_file, best_file_hv = find_best(os.path.join(directory, fitness_function))
    df_influence_seedSize_time_hv = pd.read_csv(os.path.join(directory, fitness_function, best_file_hv), sep = ',')
    df_influence_seedSize_time = pd.read_csv(os.path.join(directory, fitness_function, best_file), sep = ',')
    
    max_hv_influence_seedSize_time = df_influence_seedSize_time_hv.iloc[-1]
    print(max_hv_influence_seedSize_time)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    axs[0, 0].scatter(df_influence_seedSize_time["n_nodes"], df_influence_seedSize_time["influence"])
    axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
    axs[0, 1].scatter(df_influence_seedSize_time["n_nodes"], np.log10(df_influence_seedSize_time["time"]))
    axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 1].set_ylabel('log(1/Time)',fontsize=12)
    axs[1, 0].scatter(df_influence_seedSize_time["influence"], df_influence_seedSize_time["time"])
    axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
    axs[1, 0].set_ylabel('1/Time',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    fitness_function = "influence_seedSize_communities_time"
    best_file, best_file_hv = find_best(os.path.join(directory, fitness_function))
    df_influence_seedSize_time_communities_hv = pd.read_csv(os.path.join(directory, fitness_function, best_file_hv), sep = ',')
    df_influence_seedSize_time_communities = pd.read_csv(os.path.join(directory, fitness_function, best_file), sep = ',')
    
    max_hv_influence_seedSize_time_communities = df_influence_seedSize_time_communities_hv.iloc[-1]
    print(max_hv_influence_seedSize_time_communities)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    axs[0, 0].scatter(df_influence_seedSize_time_communities["n_nodes"], df_influence_seedSize_time_communities["influence"])
    axs[0, 0].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 0].set_ylabel('% Influenced Nodes',fontsize=12)
    axs[0, 1].scatter(df_influence_seedSize_time_communities["n_nodes"], df_influence_seedSize_time_communities["time"])
    axs[0, 1].set_xlabel('% Nodes as seed set',fontsize=12)
    axs[0, 1].set_ylabel('1/Time',fontsize=12)
    axs[1, 0].scatter(df_influence_seedSize_time_communities["influence"], df_influence_seedSize_time_communities["time"])
    axs[1, 0].set_xlabel('% Influenced Nodes',fontsize=12)
    axs[1, 0].set_ylabel('1/Time',fontsize=12)
    axs[1, 1].scatter(df_influence_seedSize_time_communities["communities"], df_influence_seedSize_time_communities["time"])
    axs[1, 1].set_xlabel('Number of Communities',fontsize=12)
    axs[1, 1].set_ylabel('1/Time',fontsize=12)

    plt.suptitle(fitness_function, fontsize=14)
    if show: 
        plt.savefig(os.path.join(new_dir, fitness_function + ".png"))
        plt.show()


    plt.scatter(df_influence_seedSize_time_communities["n_nodes"], df_influence_seedSize_time_communities["influence"], color='red', label='influence_seedSize_communities_time')
    plt.scatter(df_influence_seedSize_communities["n_nodes"], df_influence_seedSize_communities["influence"], color='green', label='influence_seedSize_communities') 
    plt.scatter(df_influence_seedSize["n_nodes"], df_influence_seedSize["influence"], color='blue', label='influence_seed')
    plt.scatter(df_influence_seedSize_time["n_nodes"], df_influence_seedSize_time["influence"], color='black', label='influence_seedSize_time')
    plt.title('Comparing fitness functions', x=0.2, y=0.5,fontsize=12,weight="bold")
    plt.ylabel('% Influenced Nodes',fontsize=12)
    plt.xlabel('% Nodes as seed set',fontsize=12)
    plt.legend()
    if show: 
        plt.savefig(os.path.join(new_dir, "all_influence_seed" + ".png"))
        plt.show()

    with open(os.path.join(new_dir, "hv.txt"), 'w') as file:

        file.write(str(max_hv_influence_seedSize) + "\n")
        file.write("\n")
        file.write(str(max_hv_influence_seedSize_communities) + "\n")
        file.write("\n")
        file.write(str(max_hv_influence_seedSize_time) + "\n")
        file.write("\n")
        file.write(str(max_hv_influence_seedSize_time_communities) + "\n")
        file.write("\n")

    res = []
    res.append(["infl_seed", max_hv_influence_seedSize["hv_influence_seed"]])
    res.append(["infl_seed_com",
                max_hv_influence_seedSize_communities["hv_influence_seed"], 
                max_hv_influence_seedSize_communities["hv_influence_communities"],
                max_hv_influence_seedSize_communities["hv_seed_communities"], 
                max_hv_influence_seedSize_communities["hv_influence_seedSize_communities"]])

    res.append(["infl_seed_time",
            max_hv_influence_seedSize_time["hv_influence_seed"], 
            None, 
            None,
            None,  
            max_hv_influence_seedSize_time["hv_influence_time"],
            max_hv_influence_seedSize_time["hv_seed_time"], 
            max_hv_influence_seedSize_time["hv_influence_seedSize_time"]])

    res.append([ "infl_seed_com_time", 
            max_hv_influence_seedSize_time_communities["hv_influence_seed"], 
            max_hv_influence_seedSize_time_communities["hv_influence_communities"],
            max_hv_influence_seedSize_time_communities["hv_seed_communities"], 
            max_hv_influence_seedSize_time_communities["hv_influence_seedSize_communities"], 
            max_hv_influence_seedSize_time_communities["hv_influence_time"],
            max_hv_influence_seedSize_time_communities["hv_seed_time"], 
            max_hv_influence_seedSize_time_communities["hv_influence_seedSize_time"], 
            max_hv_influence_seedSize_time_communities["hv_influence_seedSize_communities_time"]])

    df = pd.DataFrame(res)
    df.columns = ["fitness_function", "hv_influence_seed", "hv_influence_communities","hv_seed_communities", "hv_influence_seedSize_communities", 
                "hv_influence_time", "hv_seed_time", "hv_influence_seedSize_time", "hv_influence_seedSize_communities_time"]
    df.to_csv(os.path.join(new_dir, "hv.csv"))