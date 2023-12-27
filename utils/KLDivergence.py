import torch

from utils.df_utils.shuffle_df import shuffle_cols, shuffle_rows
from utils.df_utils.vectorise_df import vectorise_df
from utils.ts_utils.shuffle_ts import shuffle_2D_vect_ts


def split_train_test(data_df, split_params):
    seed = split_params['seed']
    
    if split_params['random_q_order']:
        data_df = shuffle_cols(data_df, shuffle_seed=seed, reset_index=False) # shuffle question order
    
    if split_params['random_s_order']:
        data_df = shuffle_rows(data_df, shuffle_seed=seed, reset_index=False) # shuffle student order

    # Separate out questions to be involved in test set as 'selected_df'
    test_q_params = split_params['test_q_params']
    selected_df, remaining_df = split_test_questions(data_df, test_q_params['q_range'], test_q_params['specific'])

    # Vectorise and shuffle selected_df
    selected_ts = vectorise_df(selected_df)
    selected_ts = shuffle_2D_vect_ts(selected_ts, shuffle_seed=seed)

    # Split the last 'e.g. test_split=30%' out as test set, the rest as train set
    num_test_entries = int(selected_ts.shape[1] * split_params['test_split'])
    num_train_entries = selected_ts.shape[1] - num_test_entries
    train_ts, test_ts = torch.split(selected_ts, [num_train_entries, num_test_entries], dim=1)

    # Concat the original remaining entries to the train set
    remaining_ts = vectorise_df(remaining_df)
    train_ts = torch.cat([train_ts, remaining_ts], 1)
    train_ts = shuffle_2D_vect_ts(train_ts, shuffle_seed=seed)

    if split_params['val_split'] != None:
        num_val_entries = int((train_ts.shape[1] * split_params['val_split']))
        num_train_entries = train_ts.shape[1] - num_val_entries
        train_ts, val_ts = torch.split(train_ts, [num_train_entries, num_val_entries], dim=1)
        return train_ts, test_ts, val_ts

    return train_ts, test_ts, None


def split_test_questions(data_df, q_range, specific):
    '''
    Separate out questions involved in test set
    if q_range and specific are both None, all questions are involved in test set
    '''
    if q_range != None:
        test_specific_questions = list(range(q_range[0], q_range[1]))

    elif specific != None:
        test_specific_questions = specific

    else:
        test_specific_questions = list(range(0, data_df.shape[1]))
        
    # selected_df = data_df[test_specific_questions]
    selected_df = data_df.iloc[:, test_specific_questions]
    remaining_df = data_df.drop(columns=test_specific_questions)
    return selected_df, remaining_df
