import yaml
import torch

from utils.get_pathname import get_data_pathname_sim
from models.M2PL import M2PL

# run model
split_config = 'synthetic-AbDif'
synthetic_config = 'AbDif'
additional = None

with open(f'config/synthetic/{synthetic_config}.yaml') as l:
    synthetic_params = yaml.safe_load(l)

my_model = M2PL(synthetic_params)
data_pathname = get_data_pathname_sim(synthetic_config)
synthetic_df, ground_truth_params = my_model.synthesise(synthetic_params, save=data_pathname)