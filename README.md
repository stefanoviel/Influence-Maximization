# Influence-Maximization
Multi-Objective optimization of Influence Maximization over graphs with evolutionary algorithms.
NSGA2 was used for the optimization of the following 3 objective funcions:
- obj 1: number of reached nodes starting from a seed set (to maximize)
- obj 2: number of nodes in the seed set (to minimize)
- obj 3: time required to converge to the optima solution (to minimize)

## Folder's Structure
```
Influence-Maximization
│   README.md
│   graph_influence.py
│   new_ea.py
│   load.py
│   override.py
│   threadpool.py
│   spread.py
│   functions.py
│     
└───plot
│   │   3d_plot.py
│   │   2d_plot.py
│   │   matrix_scatter2d.py
│
└─── ExperimentsResults
│   │
│   └─── Amazon0302
│   │
│   └─── ca-GrQ
│   │
│   └─── cego/facebook
│ 
└─── graphs
│   │ Amazon0303.txt
│   │ ca-GrQc.txt
│   │ facebook_combined.txt
│  
└─── src_OLD
    │  Previous Codes
    │  ... 
```

## Contributors
This project was developed for the university course Bio Inspyred Artificial Intelligence of the DISI department of the University of Trento.

- Elia Cunegatti MSc Computer Science Student University of Trento 
- Edoardo Schioccola MSc Computer Science Student University of Trento

The whole project has been followed and carried out under the supervision of the course holder Professor Giovanni Iacca.

