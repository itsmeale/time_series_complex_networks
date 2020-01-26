# -*- coding: utf-8 -*-

import pandas as pd


nodes = pd.read_csv("data/iterim/nodes.csv")
nodes_with_communities = pd.read_csv("data/iterim/nodes_with_communities.csv")
test = pd.read_csv("data/iterim/test.csv")

# creating test feature hash
test["hash"] = pd.Series(test.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6"]].values.tolist()).map(lambda x: ''.join(map(str, x)))

nodes = nodes.merge(
    right=nodes_with_communities,
    right_on="node_id",
    left_on="node_id"
)

test = test.merge(
    right=nodes,
    right_on="hash",
    left_on="hash",
    how="left"
)

test.to_csv("data/iterim/test.csv", index=False)
