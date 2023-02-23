import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from celf_pareto_fronts import find_best


def boxplot(directory): 
    label = directory.replace('exp1_out_', '')
    print(label)

    df_final = pd.DataFrame()

    for f in ['influence_seedSize', 'influence_communities', 'influence_time']: 
        # TODO: remove try catch once all experiment have been rerun
        try: 
            _, best_file = find_best(os.path.join(directory, f), 'hv_' + f)
        except KeyError as e : 
            print(e)
            _, best_file = find_best(os.path.join(directory, f), 'hv_influence_seed')

        df = pd.read_csv(os.path.join(directory, f, best_file), sep = ',')
        df['fitness_function '] = [f] * len(df)
        df_final = pd.concat([df_final, df], axis = 0, ignore_index=True)


    sns.boxplot(data = df_final, x='fitness_function ', y='influence')
    plt.title(label)
    plt.savefig('result_comparison/' + label + '/' + 'pf_noSeed_boxplot.png')
    plt.show()


if "__main__" == __name__: 
    for directory in os.listdir(): 
        if 'exp1_out' in directory: 
            print(directory)
            boxplot(directory)
