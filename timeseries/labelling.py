# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def close_price_variation(dataframe, over_col):
    temp_df = dataframe.copy()
    len_df = len(dataframe)
    variation_col = f"variation_{over_col}"

    temp_df[variation_col] = None
    for i in range(len_df-1):
        today_price = temp_df.loc[i, over_col]
        tomorrow_price = temp_df.loc[i + 1, over_col]
        variation = np.log(tomorrow_price) - np.log(today_price)
        temp_df.loc[i, variation_col] = variation

    return temp_df[variation_col]


def label(variation):
    if variation >= 0:
        return 1
    return -1


feature_matrix = pd.read_csv("data/iterim/feature_matrix_with_prices_and_communities.csv", dtype=np.float)
feature_matrix["raw_price_variation"] = pd.to_numeric(close_price_variation(feature_matrix, over_col="cp"))
feature_matrix["smooth_price_variation"] = pd.to_numeric(close_price_variation(feature_matrix, over_col="cpj"))

community_label = feature_matrix.loc[:, ["community", "raw_price_variation", "smooth_price_variation"]]
community_label = community_label.dropna()

community_raw_label = community_label.groupby("community")["raw_price_variation"].mean().reset_index()
community_smooth_label = community_label.groupby("community")["smooth_price_variation"].mean().reset_index()

community_raw_label["raw_price_label"] = community_raw_label["raw_price_variation"].apply(label)
community_smooth_label["smooth_price_label"] = community_smooth_label["smooth_price_variation"].apply(label)

cols = ["t", "community", "X1", "X2", "X3", "X4", "X5", "X6", "raw_price_label", "smooth_price_label"]

feature_matrix_with_labels = (
    feature_matrix
    .merge(
        right=community_raw_label,
        right_on="community",
        left_on="community"
    )
    .merge(
        right=community_smooth_label,
        right_on="community",
        left_on="community"
    )
    .loc[:, cols]
)

feature_matrix_with_labels.to_csv("data/iterim/feature_matrix_with_labels.csv", index=False)
