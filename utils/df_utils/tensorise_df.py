import torch


def tensorise_df(data_df):
    return torch.tensor(data_df.values)
