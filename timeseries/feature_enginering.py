# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


DATA_FOLDER = "data/raw/{}"
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
		if i-1 < 0 or np.isnan(temp_df.loc[i-1, over]): continue
		temp_df.loc[i, new_col] = (temp_df.loc[i, over] / temp_df.loc[i-1, over] - 1)

	return temp_df[new_col]


original_series = pd.read_csv(DATA_FOLDER.format(RAW_DATA_FILE), sep="\t")

original_series.columns = ["date", "cp"]
original_series["d"] = original_series.index + 1  # starts the index from 1
original_series["cpj"] = original_series["cp"].rolling(window=J).mean()

original_series[MAq_short] = original_series.cpj.rolling(Q_short).mean()
original_series[MAq_long] = original_series.cpj.rolling(Q_long).mean()

original_series["f1"] = original_series.cpj / original_series[MAq_short] - 1
original_series["f2"] = original_series.cpj / original_series[MAq_long] - 1
original_series["f3"] = gradient(original_series, over=MAq_short)
original_series["f4"] = gradient(original_series, over=MAq_long)
original_series["f5"] = (original_series.cpj - original_series.cpj.rolling(window=Q_short).min()) / (original_series.cpj.rolling(window=Q_short).max() - original_series.cpj.rolling(window=Q_short).min())
original_series["f6"] = (original_series.cpj - original_series.cpj.rolling(window=Q_long).min()) / (original_series.cpj.rolling(window=Q_long).max() - original_series.cpj.rolling(window=Q_long).min())

original_series.to_csv("temp.csv", index=False)
