import pandas as pd


def drop_nan_rows(data_df: pd.DataFrame) -> pd.DataFrame:
    '''drop rows with nan values, index retained'''
    return data_df.dropna(axis=0, how='any')
