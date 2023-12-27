import torch
import pandas as pd

from utils.df_utils.tensorise_df import tensorise_df


def vectorise_df(data_df: pd.DataFrame) -> torch.Tensor:
    '''
    Return torch.Tensor with dtype torch.int32
    '''
    data_ts = tensorise_df(data_df)
    row_idx_ts, col_idx_ts = get_row_col_idx_ts(data_df)

    unstacked_ts = unstack_ts(data_ts)
    unstacked_row_idx_ts = unstack_ts(row_idx_ts)
    unstacked_col_idx_ts = unstack_ts(col_idx_ts)

    vectorised_ts = stack_ts(unstacked_ts, unstacked_row_idx_ts, unstacked_col_idx_ts)
    vectorised_ts = drop_nan_entries(vectorised_ts) # remove entries with nan values
    vectorised_ts = vectorised_ts.type(torch.int)
    return vectorised_ts


def get_row_col_idx_ts(data_df):
    row_idx = data_df.index
    col_idx = data_df.columns

    row_idx_ts = torch.tensor(row_idx).repeat(len(col_idx), 1).T.type(torch.int)
    col_idx_ts = torch.tensor(col_idx).repeat(len(row_idx), 1).type(torch.int)
    return row_idx_ts, col_idx_ts


def unstack_ts(data_ts: torch.Tensor) -> torch.Tensor:
    return data_ts.reshape(-1)


def stack_ts(data_ts, row_idx_ts, col_idx_ts):
    return torch.stack((data_ts, row_idx_ts, col_idx_ts), dim=0)


def drop_nan_entries(data_ts: torch.Tensor) -> torch.Tensor:
    return torch.clone(data_ts.T[~torch.any(data_ts.isnan(),dim=0)].T) # remove entries containing nan values
