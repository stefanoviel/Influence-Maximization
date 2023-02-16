import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


for subd in os.listdir('result_comparison'): 

    df_avg = pd.read_csv('result_comparison/'+ subd + '/avg_hv.csv')
    df_std = pd.read_csv('result_comparison/'+ subd + '/std_hv.csv')

    df_avg = df_avg.iloc[:, :5]
    df_std = df_std.iloc[:, :5]
    df_avg = df_avg.loc[:, ~df_avg.columns.str.contains('^Unnamed')]
    df_std = df_std.loc[:, ~df_std.columns.str.contains('^Unnamed')]

    df_new = []
    for (index, row), (index1, row1) in zip(df_avg.iterrows(), df_std.iterrows()):
        df_new.append([row["fitness_function"], "hv_influence_seed" , row["hv_influence_seed"], row1["hv_influence_seed"]])
        df_new.append([row["fitness_function"], "hv_influence_seedSize_time", row["hv_influence_seedSize_time"], row1["hv_influence_seedSize_time"]])
        df_new.append([row["fitness_function"], "hv_influence_seedSize_communities", row["hv_influence_seedSize_communities"], row1["hv_influence_seedSize_communities"]])

    df_new = pd.DataFrame(df_new)
    df_new.columns = ['fitness_function', 'hv_measure', 'hypervolume values', 'hypervolume std']
    
    # set width of bar
    barWidth = 0.2
    f, ax = plt.subplots(1, 3, figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(1)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    for i in range(3): 
        print(df_new.loc[df_new["fitness_function"] == "infl_seed"]["hypervolume values"].iloc[i])
        ax[i].bar(br1, df_new.loc[df_new["fitness_function"] == "infl_seed"]["hypervolume values"].iloc[i], color ='r', width = barWidth,
                edgecolor ='grey', label ='infl_seed')
        ax[i].bar(br2, df_new.loc[df_new["fitness_function"] == "infl_seed_com"]["hypervolume values"].iloc[i], color ='b', width = barWidth,
                edgecolor ='grey', label ='infl_seed_com')
        ax[i].bar(br3, df_new.loc[df_new["fitness_function"] == "infl_seed_time"]["hypervolume values"].iloc[i], color ='g', width = barWidth,
                edgecolor ='grey', label ='infl_seed_time')
        ax[i].bar(br4, df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]["hypervolume values"].iloc[i], color ='yellow', width = barWidth,
                edgecolor ='grey', label ='infl_seed_com_time')

        # Adding std
        # ax[i].errorbar(x = br1, y = df_new.loc[df_new["fitness_function"] == "infl_seed"]["hypervolume values"].iloc[i], 
        #     yerr=df_new.loc[df_new["fitness_function"] == "infl_seed"]['hypervolume std'].iloc[i], fmt='none', c= 'black', capsize = 2)
        # ax[i].errorbar(x = br2, y = df_new.loc[df_new["fitness_function"] == "infl_seed_com"]["hypervolume values"].iloc[i],
        #     yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_com"]['hypervolume std'].iloc[i], fmt='none', c= 'black', capsize = 2)
        # ax[i].errorbar(x = br3, y = df_new.loc[df_new["fitness_function"] == "infl_seed_time"]["hypervolume values"].iloc[i], 
        #     yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_time"]['hypervolume std'].iloc[i], fmt='none', c= 'black', capsize = 2)
        # ax[i].errorbar(x = br4, y = df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]["hypervolume values"].iloc[i],
        #     yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]['hypervolume std'].iloc[i], fmt='none', c= 'black', capsize = 2)

        # Adding Xticks
#     f.xlabel('fitness functions', fontweight ='bold', fontsize = 15, labelpad=20)
#     ax[0].ylabel('hypervolume values', fontweight ='bold', fontsize = 15)
#     f.xticks([r + barWidth for r in range(len(df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]))], ['hv_influence_seed', 'hv_influence_seedSize_time', 'hv_influence_seedSize_communities' ])
    
    plt.legend()
    plt.title(subd)
    plt.savefig('result_comparison/'+ subd + '/avg_hbar.png',  bbox_inches='tight', dpi=300)
    plt.show()

    break