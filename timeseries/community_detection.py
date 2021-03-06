# -*- coding: utf-8 -*-

import pandas as pd
from networkx import Graph
from networkx.algorithms.community import greedy_modularity_communities


ITERIM_DATA_FOLDER = "data/iterim/{}"
NETWORK_FILE = "network_edges.csv"
FEATURE_MATRIX_FILE = "feature_matrix.csv"


network_edges = pd.read_csv(ITERIM_DATA_FOLDER.format(NETWORK_FILE))
feature_matrix = pd.read_csv(ITERIM_DATA_FOLDER.format(FEATURE_MATRIX_FILE))

network_edges = network_edges.loc[:, ["source", "target"]].values.tolist()

graph = Graph()
graph.add_edges_from(network_edges)

communities = greedy_modularity_communities(graph)

nodes_communities = list()
feature_matrix["community"] = None
for community_id, community in enumerate(communities):
    for node in community:
        feature_matrix.loc[feature_matrix.node_id == node, "community"] = community_id
        nodes_communities.append((node, community_id))

nodes_communities_df = pd.DataFrame(nodes_communities, columns=["node_id", "community"])
nodes_communities_df.to_csv(ITERIM_DATA_FOLDER.format("nodes_with_communities.csv"), index=False)

train = pd.read_csv("data/iterim/train.csv")
train = train.loc[:, ["t", "cp", "cpj"]]

feature_matrix_with_prices = feature_matrix.merge(
    right=train,
    right_on="t",
    left_on="t"
)

feature_matrix_with_prices.to_csv("data/iterim/feature_matrix_with_prices_and_communities.csv", index=False)
