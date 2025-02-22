# Allows to run all the fitness function together
# no_seed needs to be passed if we want to run the set of fitness function where the seed set size is fixed

import json
import sys
import time
import subprocess

fit_func = [
    "influence_seedSize",
    "influence_seedSize_time",
    "influence_seedSize_communities",
    "influence_seedSize_communities_time"
]

if 'no_seed' in sys.argv: 
    print('no seed')
    fit_func = [
        "influence_time",
        "influence_communities",
        "influence_communities_time"
    ]

for f_f in fit_func:
    print("executing", f_f)

    with open("config.json", "r") as jsonFile:
        data = json.load(jsonFile)

    # number of underscore + 1 = number objectives
    data["script_args"]["elements_objective_function"] = f_f

    with open("config.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

    bashCommand = "python influence_maximization.py --config config.json"
    process = subprocess.Popen(bashCommand.split(), stdin=None, stdout=None, stderr=None, close_fds=True)
    time.sleep(5)
