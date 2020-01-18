# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import norm
from math import ceil

RAW_DATA_FOLDER = "data/raw/{}"
ITERIM_DATA_FOLDER = "data/iterim/{}"
RAW_DATA_FILE = "BVSP_orig.txt"

J = 5
Q_short = 10
Q_long = 50
MAq_short = f"maq_{Q_short}"
MAq_long = f"maq_{Q_long}"


def gradient(dataframe, over):
    temp_df = dataframe.copy()
    new_col = "temp_col"
    temp_df[new_col] = np.nan
    for i in range(len(temp_df)):
        if i-1 < 0 or np.isnan(temp_df.loc[i-1, over]):
            continue
        temp_df.loc[i, new_col] = (temp_df.loc[i, over] / temp_df.loc[i-1, over] - 1)
    return temp_df[new_col]


def feature_engineering():
    original_series = pd.read_csv(RAW_DATA_FOLDER.format(RAW_DATA_FILE), sep="\t")
    original_series.columns = ["date", "cp"]

    # creating features
    original_series["d"] = original_series.index + 1  # starts the index from 1
    original_series["cpj"] = original_series["cp"].rolling(window=J).mean()
    original_series[MAq_short] = original_series.cpj.rolling(Q_short).mean()
    original_series[MAq_long] = original_series.cpj.rolling(Q_long).mean()
    original_series["f1"] = original_series.cpj / original_series[MAq_short] - 1
    original_series["f2"] = original_series.cpj / original_series[MAq_long] - 1
    original_series["f3"] = gradient(original_series, over=MAq_short)
    original_series["f4"] = gradient(original_series, over=MAq_long)
    original_series["f5"] = (
        (original_series.cpj - original_series.cpj.rolling(window=Q_short).min()) /
        (original_series.cpj.rolling(window=Q_short).max() - original_series.cpj.rolling(window=Q_short).min())
    )
    original_series["f6"] = (
        (original_series.cpj - original_series.cpj.rolling(window=Q_long).min()) /
        (original_series.cpj.rolling(window=Q_long).max() - original_series.cpj.rolling(window=Q_long).min())
    )
    original_series = original_series.dropna()
    original_series["t"] = list(range(1, len(original_series) + 1))

    # creating z score and f(z score) for features f1, f2, f3, f4, f5, f6
    features = 6
    for feature in range(1, features + 1):
        feature_z = f"z{feature}"
        feature_f = f"f{feature}"
        feature_fz = f"fz{feature}"
        feature_x = f"X{feature}"
        original_series[feature_z] = (
            (original_series[feature_f] - original_series[feature_f].mean()) /
            original_series[feature_f].std()
        )
        original_series[feature_fz] = original_series[feature_z].apply(norm.cdf)
        original_series[feature_x] = original_series[feature_fz].apply(lambda x: ceil(x/.2) / (1/.2))

    # separing data between train and test datasets and saving as csv
    train = original_series.iloc[:5638, :]
    test = original_series.iloc[5638:, :]
    train.to_csv(ITERIM_DATA_FOLDER.format("train.csv"), index=False)
    test.to_csv(ITERIM_DATA_FOLDER.format("test.csv"), index=False)


if __name__ == "__main__":
    feature_engineering()
