# -*- coding: utf-8 -*-

import pandas as pd


def close_price_variation(dataframe, over_col):
    pass

feature_matrix = pd.read_csv("data/iterim/feature_matrix_with_communities.csv")
train = pd.read_csv("data/iterim/train.csv")
train = train.loc[:, ["t", "cp", "cpj"]]

feature_matrix_with_prices = feature_matrix.merge(
    right=train,
    right_on="t",
    left_on="t"
)

