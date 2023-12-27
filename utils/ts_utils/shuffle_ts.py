import random
import torch


def shuffle_1D_vect_ts(vectorised_ts, shuffle_seed=1000):
    col_idxs = list(range(vectorised_ts.shape[0]))
    random.seed(shuffle_seed)
    random.shuffle(col_idxs)
    vectorised_ts = torch.clone(vectorised_ts[torch.tensor(col_idxs)])
    return vectorised_ts


def shuffle_2D_vect_ts(vectorised_ts, shuffle_seed=1000):
    col_idxs = list(range(vectorised_ts.shape[1]))
    random.seed(shuffle_seed)
    random.shuffle(col_idxs)
    vectorised_ts = torch.clone(vectorised_ts[:, torch.tensor(col_idxs)])
    return vectorised_ts
