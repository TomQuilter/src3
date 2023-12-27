import pandas as pd


def combine_df(df_arr):
    '''
    df concatenated according to row index, df with no data points at the row index becomes 'nan'
    '''
    return pd.concat(df_arr, axis=1)
