import json
import time
import subprocess

fit_func = [
    "influence_seedSize",
    "influence_seedSize_time",
    "influence_seedSize_communities",
    "influence_seedSize_communities_time"
]

for f_f in fit_func:
    print("executing", f_f)

    with open("config.json", "r") as jsonFile:
        data = json.load(jsonFile)

    # number of underscore + 1 = number objectives
    data["script_args"]["no_obj"] = f_f.count('_') + 1
    data["script_args"]["elements_objective_function"] = f_f

    with open("config.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

    bashCommand = "python influence_maximization.py --config config.json"
    process = subprocess.Popen(bashCommand.split(), stdin=None, stdout=None, stderr=None, close_fds=True)
    time.sleep(5)
    # output, error = process.communicate()

    # if error != None:
    #     raise("Error: ", error)
