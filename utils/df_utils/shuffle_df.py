import numpy as np

from utils.df_utils.reset_index_df import reset_row_index, reset_col_index


def shuffle_rows(df, shuffle_seed=1000, reset_index=True):
    shuffled_df = df.copy()
    shuffled_df = shuffled_df.sample(frac=1, axis=0, random_state=np.random.RandomState(shuffle_seed))
    if reset_index:
        shuffled_df = reset_row_index(shuffled_df)
    return shuffled_df


def shuffle_cols(df, shuffle_seed=1000, reset_index=True):
    shuffled_df = df.copy()
    shuffled_df = shuffled_df.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed))
    # orig_col_order = shuffled_df.columns
    if reset_index:
        shuffled_df = reset_col_index(shuffled_df)
    return shuffled_df
