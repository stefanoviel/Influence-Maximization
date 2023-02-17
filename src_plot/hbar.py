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
    fig = plt.subplots(figsize =(12, 9))
    

    # Set position of bar on X axis
    br1 = np.arange(len(df_new.loc[df_new["fitness_function"] == "infl_seed"]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    

    # Make the plot
    plt.bar(br1, df_new.loc[df_new["fitness_function"] == "infl_seed"]["hypervolume values"], color ='r', width = barWidth,
            edgecolor ='grey', label ='base')
    plt.bar(br2, df_new.loc[df_new["fitness_function"] == "infl_seed_com"]["hypervolume values"], color ='b', width = barWidth,
            edgecolor ='grey', label ='communities')
    plt.bar(br3, df_new.loc[df_new["fitness_function"] == "infl_seed_time"]["hypervolume values"], color ='g', width = barWidth,
            edgecolor ='grey', label ='time')
    plt.bar(br4, df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]["hypervolume values"], color ='yellow', width = barWidth,
            edgecolor ='grey', label ='#communities & time')

    # Adding std
    plt.errorbar(x = br1, y = df_new.loc[df_new["fitness_function"] == "infl_seed"]["hypervolume values"], 
        yerr=df_new.loc[df_new["fitness_function"] == "infl_seed"]['hypervolume std'], fmt='none', c= 'black', capsize = 2)
    plt.errorbar(x = br2, y = df_new.loc[df_new["fitness_function"] == "infl_seed_com"]["hypervolume values"], 
        yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_com"]['hypervolume std'], fmt='none', c= 'black', capsize = 2)
    plt.errorbar(x = br3, y = df_new.loc[df_new["fitness_function"] == "infl_seed_time"]["hypervolume values"], 
        yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_time"]['hypervolume std'], fmt='none', c= 'black', capsize = 2)
    plt.errorbar(x = br4, y = df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]["hypervolume values"], 
        yerr=df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]['hypervolume std'], fmt='none', c= 'black', capsize = 2)

    # Adding Xticks
    plt.xlabel('Type of hypervolume', fontweight ='bold', fontsize = 15, labelpad=20)
    plt.ylabel('Hypervolume Values', fontweight ='bold', fontsize = 15, labelpad= 20)
    plt.xticks([r + barWidth for r in range(len(df_new.loc[df_new["fitness_function"] == "infl_seed_com_time"]))], ['HV influence seed size', 'HV influence seed size \ntime', 'HV influence seed size \n#communities' ])
    
    plt.legend()
    plt.title(subd)
    plt.savefig('result_comparison/'+ subd + '/avg_hbar.png',  bbox_inches='tight', dpi=300)
    plt.show()

    break