import torch


def count_ones_df(data_df):
    data_ts = torch.tensor(data_df.values)
    return count_ones_ts(data_ts)


def count_ones_ts(data_ts):
    num_ones = torch.sum(data_ts)
    num_entries = torch.numel(data_ts)
    return num_ones/num_entries
