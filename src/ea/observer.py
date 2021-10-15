import pandas as pd
def ea_observer(archiver, population_file) :
    
    print("OBSERVERRRR \n  OBSERVERRRR \n OBSERVERRRR \n OBSERVERRRR \n OBSERVERRRR \n OBSERVERRRR \n OBSERVERRRR \n")

    
    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    time = []
    a = []
    for item in archiver:
        nodes.append(str(item[0]))
        influence.append(item[1])
        n_nodes.append(item[2])
        time.append(item[3])
        a.append(len(item[0]))

    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["time"] = time
    df["nodes"] = nodes
    df["a"] = a
    df.to_csv(population_file+".csv", sep=",", index=False)
    # #find the longest individual
    # max_length = len(max(archiver, key=lambda x : len(x.candidate)).candidate)

    # with open(archiver_file, "w") as fp :
    #     # header, of length equal to the maximum individual length in the archiver
    #     fp.write("n_nodes,influence,n_simulation")

    #     for i in range(0, max_length) : fp.write(",n%d" % i)
    #     fp.write("\n")

    #     # and now, we write stuff, individual by individual
    #     for individual in  archiver :

    #         # check if fitness is an iterable collection (e.g. a list) or just a single value
    #         if hasattr(individual.fitness, "__iter__") :
    #             fp.write("%d,%.4f,%d" % (1.0 / individual.fitness[1], individual.fitness[0],  (1.0 / individual.fitness[2])))
    #         else :
    #             fp.write("%d,%.4f" % (len(set(individual.candidate)), individual.fitness))

    #         for node in individual.candidate :
    #             fp.write(",%d" % node)

    #         for i in range(len(individual.candidate), max_length - len(individual.candidate)) :
    #             fp.write(",")

    #         fp.write("\n")