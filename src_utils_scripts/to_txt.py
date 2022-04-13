import os
import argparse
import pandas as pd


def read_arguments():
    """
	Parameters for the upscaling process process.
	"""
    parser = argparse.ArgumentParser(
        description='Upscalinf algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--path', default='None', type=str,
                        help='Path of your file')

    # Upscaling Parameters
    parser.add_argument('--format', type=str, default='.edges',
                        help='Dataset file format')  
    parser.add_argument('--column1', default='node1',
                        type=str, help='Column 1 name')
    parser.add_argument('--column2', default='node2',
                        type=str, help='Column 2 name')

    args = parser.parse_args()
    args = vars(args)

    return args


if __name__ == '__main__':

    args = read_arguments()
    name = (os.path.basename(args["path"]))
    name = name.replace(args["format"],"")

    df = pd.read_csv(args["path"], sep=" ",index_col=False)

    n1 = df[args["column1"]].to_list()
    n2 = df[args["column2"]].to_list()


    n1 = [item for item in n1]
    n2 = [item for item in n2]


    if 0 not in n1 and 0 not in n2:
        n1 = [item-1 for item in n1]
        n2 = [item-1 for item in n2] 

    text = []
    for i in range(len(n1)):
        f = "{0} {1}".format(n1[i],n2[i])
        text.append(f) 
    with open('{0}.txt'.format(name), "w") as outfile:
            outfile.write("\n".join(text))
            