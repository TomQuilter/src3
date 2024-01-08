import yaml

from utils.get_pathname import get_pathname
from utils.process_3papers import process_3papers
from utils.split_train_test import split_train_test

from models.AbilityDifficulty import AbilityDifficulty
from models.Interactive import Interactive
from models.InteractiveVI import InteractiveVI
from models.AbilityDifficultyVariationalInference import AbilityDifficultyVariationalInference
 
split_config = 'default'

# Git working :):):):):):): both ways!:)
#model_config = 'AbDif'
# model_config = 'AbDifVI'
#model_config = 'Int2'
model_config = 'Int2VI'
      
print("### Process Beginning Check ###") 
  
with open(f'config/split/{split_config}.yaml') as f:
    split_params = yaml.safe_load(f)
 
with open(f'config/model/{model_config}.yaml') as o:
    model_params = yaml.safe_load(o)

additional = None
raw_pathname, plot_pathname = get_pathname(split_config, model_config, additional)

data_df, meta_df = process_3papers()
train_ts, test_ts, val_ts = split_train_test(data_df, split_params)
 
if model_config == 'AbDif':
    #my_model = AbilityDifficultyVariationalInference(model_params)
    my_model = AbilityDifficulty(model_params)
elif  model_config == 'Int2':
    my_model = Interactive(model_params)
elif  model_config == 'AbDifVI':
    my_model = AbilityDifficultyVariationalInference(model_params)
elif  model_config == 'Int2VI':
    my_model = InteractiveVI(model_params)
  
results = my_model.run(train_ts, test_ts, val_ts, data_df, meta_df=None, save=raw_pathname, plot=plot_pathname)
print(results['res']['acc'], results['res']['conf'], results['hyperparams']['iters'])


