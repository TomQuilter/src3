import torch

from utils.get_pathname import get_pathname, get_pathname_sim, get_data_pathname_sim


def load_result(split_config, model_config, additional):
    raw_pathname, _ = get_pathname(split_config, model_config, additional)
    results = torch.load(raw_pathname)
    return results


def load_result_sim(split_config, model_config, synthetic_config, additional):
    raw_pathname, _ = get_pathname_sim(split_config, model_config, synthetic_config, additional)
    results = torch.load(raw_pathname)
    return results


def load_data_sim(synthetic_config, additional):
    raw_pathname = get_data_pathname_sim(synthetic_config, additional)
    data = torch.load(raw_pathname)
    return data
