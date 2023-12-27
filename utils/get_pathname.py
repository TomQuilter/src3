def get_pathname(split_config, model_config, additional=None):
    filename = f'split_{split_config}_model_{model_config}'
    if additional:
        filename = f'{filename}_{additional}'
    raw_pathname = f'results/raw/{filename}.pt'
    plot_pathname = f'results/plots/{filename}.png'
    return raw_pathname, plot_pathname


def get_pathname_sim(split_config, model_config, synthetic_config, additional=None):
    filename = f'split_{split_config}_model_{model_config}_synthetic_{synthetic_config}'
    if additional:
        filename = f'{filename}_{additional}'
    raw_pathname = f'results/synthetic/raw/{filename}.pt'
    plot_pathname = f'results/synthetic/plots/{filename}.png'
    return raw_pathname, plot_pathname

def get_data_pathname_sim(synthetic_config, additional=None):
    filename = f'synthetic_model_{synthetic_config}.pt'
    if additional:
        filename = f'{filename}_{additional}'
    raw_pathname = f'data/synthetic/{filename}.pt'
    return raw_pathname
