import pandas as pd
import numpy as np
import os



def get_file_hv(directory): 
    """
    From each file extract only all the hv corresponding to the generation that achieved the best results on hv_influence_seed
    compute the average and std on those values among all the hv elements
    """
    all_hv = []
    for file in os.listdir(directory): 
        if 'hv' in file: 
            df = pd.read_csv(os.path.join(directory, file), sep = ',')        
            # get the hvs of the last PF of each run (i.e. the one in the archive)
            all_hv.append(df.iloc[-1].to_list())
            col = df.columns

    res = pd.DataFrame(all_hv)
    res.columns = col
    return res


if "__main__" == __name__: 

    for directory in os.listdir(): 
        if 'exp1_out' in directory: 
            try: 
                os.mkdir(os.path.join('result_comparison', directory.replace('exp1_out_', ''))) 
            except OSError as error: 
                print(error) 

            for fitness_function in os.listdir(directory): 
                if 'seedSize' in fitness_function: 

                    try: 
                        os.mkdir(os.path.join('result_comparison', directory.replace('exp1_out_', ''), 'hvs')) 
                    except OSError as error: 
                        print(error) 

                    print(directory, fitness_function)
                    df = get_file_hv(os.path.join(directory, fitness_function))
                    df = df.drop(columns=['generation'])
                    df.to_csv(os.path.join('result_comparison', directory.replace('exp1_out_', ''), 'hvs', fitness_function + '_hvs.csv'))

