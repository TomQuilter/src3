import pandas as pd

from utils.df_utils.threshold_df import threshold_df
from utils.df_utils.binarise_df import binarise_df
from utils.df_utils.reset_index_df import reset_col_index, reset_row_index


def preprocess_data(data_df: pd.DataFrame, meta_df: pd.DataFrame):
    max_scores_df = meta_df.loc['Max'].astype(float)
    thres_df = threshold_df(data_df, max_scores_df)
    bin_df = binarise_df(thres_df, max_scores_df)

    processed_df = reset_row_index(bin_df)
    processed_df = reset_col_index(processed_df)

    processed_meta_df = reset_col_index(meta_df)
    return processed_df.copy(), processed_meta_df.copy()
