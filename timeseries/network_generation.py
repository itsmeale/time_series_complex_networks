# -*- coding: utf-8 -*-

import pandas as pd


ITERIM_DATA_FOLDER = "data/iterim/{}"
TRAIN_DATA_FILE = "train.csv"


train = pd.read_csv(ITERIM_DATA_FOLDER.format(TRAIN_DATA_FILE))
train = train.loc[:, ["t", "X1", "X2", "X3", "X4", "X5", "X6"]]


feature_matrix = train.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6"]]
feature_matrix["hash"] = pd.Series(feature_matrix.values.tolist()).map(lambda x: ''.join(map(str, x)))

feature_matrix_nodes = feature_matrix.loc[:, ['hash']].drop_duplicates().copy()
feature_matrix_nodes["node_id"] = list(range(1, len(feature_matrix_nodes) + 1))

for idx, row in feature_matrix.iterrows():
    node_id = feature_matrix_nodes[feature_matrix_nodes.hash == row.hash].node_id.values[0]
    feature_matrix.loc[idx, "node_id"] = int(node_id)

feature_matrix["t"] = list(range(1, len(feature_matrix) + 1))
feature_matrix = feature_matrix.drop(columns=["hash"])

feature_matrix_nodes.to_csv(ITERIM_DATA_FOLDER.format("nodes.csv"), index=False)
feature_matrix.to_csv(ITERIM_DATA_FOLDER.format("feature_matrix.csv"), index=False)

graph_edges = list()

for i in range(len(feature_matrix)-1):
    actual_node_id = feature_matrix.loc[i, "node_id"]
    next_node_id = feature_matrix.loc[i+1, "node_id"]

    if actual_node_id == next_node_id:
        continue

    graph_edges.append((actual_node_id, next_node_id, 1))


graph_edges = pd.DataFrame(graph_edges, columns=["source", "target", "weight"])
graph_edges = graph_edges.groupby(["source", "target"])["weight"].sum().reset_index()
graph_edges.to_csv(ITERIM_DATA_FOLDER.format("network_edges.csv"), index=False)
