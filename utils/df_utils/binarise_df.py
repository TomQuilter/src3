import pandas as pd


def binarise_df(data_df: pd.DataFrame, max_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores >= max_score/2 binarised to 1, else 0
    """
    for col in data_df:
        max_score = max_scores_df[col]
        data_df.loc[data_df[col] < max_score/2, col] = 0
        data_df.loc[data_df[col] >= max_score/2, col] = 1

    return data_df.copy()


# def binarise_PS_meta(PS_meta):
#     PS_meta.loc[PS_meta == 1.0] = 0
#     PS_meta.loc[PS_meta > 1.0] = 1
#     return PS_meta.copy()