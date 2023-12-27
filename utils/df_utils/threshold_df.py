import pandas as pd


def threshold_df(data_df: pd.DataFrame, max_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Threshold scores above max to max; scores below 0 to 0
    """
    for col in data_df:
        max_score = max_scores_df[col]
        data_df.loc[data_df[col] > max_score, col] = max_score
        data_df.loc[data_df[col] < 0, col] = 0
    return data_df.copy()
