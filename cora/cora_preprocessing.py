import os
import pandas as pd
import sys

sys.path.insert(0, '')
from src.load import read_graph

g = pd.read_csv('labelled_graphs/cora/cora_edges', sep= '\t', header=None,  names=["target", "source"])
filename = "graphs/cora_cites.txt"
name = (os.path.basename(filename))
G = read_graph(filename)
labels = pd.read_csv('labelled_graphs/cora/cora_labels', sep= '\t', header=None)

new_labels = pd.DataFrame()
new_labels['paper'] = labels.iloc[:, 0]
new_labels['label'] = labels.iloc[:, -1]

nodes = list(G.nodes())
processed_labels = pd.DataFrame()

new_labels['paper'] = new_labels['paper'].apply(lambda x : nodes.index(x))
g['source'] = g['source'].apply(lambda x : nodes.index(x))
g['target'] = g['target'].apply(lambda x : nodes.index(x))

new_labels.to_csv('labelled_graphs/cora/cora_labels_proc', sep= '\t', header=None, index=False)
g.to_csv('labelled_graphs/cora/cora_edges_proc', sep= '\t', header=None, index=False)
