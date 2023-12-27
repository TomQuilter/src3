import torch
import numpy as np

from utils.get_pathname import get_pathname
from utils.process_3papers import process_3papers
from utils.parse_paper import parse_paper
from utils.preprocess_data import preprocess_data
from utils.split_train_test import split_train_test
from utils.load_result import load_result

from utils.df_utils.shuffle_df import shuffle_cols, shuffle_rows
from utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix, precision_recall

data_df, meta_df = process_3papers()
print(data_df.shape)
# exam_df1, meta_df1 = parse_paper('new1')
# data_df, meta_df = preprocess_data(exam_df1, meta_df1)

def run(data_df, seed):
    data_df = shuffle_cols(data_df, shuffle_seed=seed, reset_index=False) # shuffle question order
    data_df = shuffle_rows(data_df, shuffle_seed=seed, reset_index=False) # shuffle student order
    data_ts = torch.tensor(data_df.values)

    first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_quadrants(data_ts, 0.5, 0.5)
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    single_param_acc = calc_single_param_acc(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, rng)
    student_ability_acc = calc_student_ability_acc(train_student_ts, test_ts, rng)
    question_difficulty_acc = calc_question_difficulty_acc(train_question_ts, test_ts, rng)

    return single_param_acc, student_ability_acc, question_difficulty_acc

def calc_single_param_acc(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, rng):
    # calculate param
    total = torch.sum(first_quadrant_ts.isnan()==0) + torch.sum(train_question_ts.isnan()==0) + torch.sum(train_student_ts.isnan()==0)
    total_correct = torch.nansum(first_quadrant_ts) + torch.nansum(train_question_ts) + torch.nansum(train_student_ts)
    single_param = total_correct/total
    
    # calculate probit
    test_ts = test_ts.reshape(-1)
    filtered_test_ts = test_ts[~test_ts.isnan()]
    probit = torch.clone(filtered_test_ts)
    probit[~probit.isnan()] = single_param

    # predict
    predictions = torch.bernoulli(probit, generator=rng)

    # performance
    acc = torch.sum(torch.eq(filtered_test_ts, predictions)) / torch.numel(filtered_test_ts)
    # conf_matrix = calc_conf_matrix(filtered_test_ts, predictions)
    # precision, recall = precision_recall(filtered_test_ts, predictions)

    return acc*100

def calc_student_ability_acc(train_student_ts, test_ts, rng):
    # calculate param
    total = torch.sum(train_student_ts.isnan()==0, dim=1)
    total_correct = torch.nansum(train_student_ts, dim=1) # total ones for each student
    student_ability = total_correct/total
    
    # calculate probit
    probit = student_ability.repeat(test_ts.shape[1], 1).T

    # predict
    predictions = torch.bernoulli(probit, generator=rng)

    # performance
    total_correct = torch.sum(torch.eq(test_ts, predictions))
    total = torch.sum(test_ts.isnan()==0)
    acc = total_correct/total
    # conf_matrix = calc_conf_matrix(filtered_test_ts, predictions)
    # precision, recall = precision_recall(filtered_test_ts, predictions)
    
    return acc*100


def calc_question_difficulty_acc(train_student_ts, test_ts, rng):
    # calculate param
    total = torch.sum(train_student_ts.isnan()==0, dim=0)
    total_correct = torch.nansum(train_student_ts, dim=0)
    question_difficulty = total_correct/total
    
    # calculate probit
    probit = question_difficulty.repeat(test_ts.shape[0], 1)

    # predict
    predictions = torch.bernoulli(probit, generator=rng)

    # performance
    total_correct = torch.sum(torch.eq(test_ts, predictions))
    total = torch.sum(test_ts.isnan()==0)
    acc = total_correct/total
    # conf_matrix = calc_conf_matrix(filtered_test_ts, predictions)
    # precision, recall = precision_recall(filtered_test_ts, predictions)
    
    return acc*100


def split_to_quadrants(data_ts, student_split, question_split):
    S, Q = data_ts.shape[0], data_ts.shape[1]
    training_students = int(S * student_split)
    training_questions = int(Q * question_split)

    upper_half_ts, lower_half_ts = torch.split(data_ts, [training_students, S-training_students], dim=0)
    first_quadrant_ts, train_question_ts = torch.split(upper_half_ts, [training_questions, Q-training_questions], dim=1)
    train_student_ts, test_ts = torch.split(lower_half_ts, [training_questions, Q-training_questions], dim=1)

    return first_quadrant_ts, train_question_ts, train_student_ts, test_ts

SP_acc_arr = []
SA_acc_arr = []
QD_acc_arr = []
for seed in range(1000, 1010):
    SP_acc, SA_acc, QD_acc = run(data_df, seed)
    SP_acc_arr.append(SP_acc)
    SA_acc_arr.append(SA_acc)
    QD_acc_arr.append(QD_acc)
    print(SP_acc, SA_acc, QD_acc)

SP_acc_mean, SP_acc_std = np.mean(SP_acc_arr), np.std(SP_acc_arr)
SA_acc_mean, SA_acc_std = np.mean(SA_acc_arr), np.std(SA_acc_arr)
QD_acc_mean, QD_acc_std = np.mean(QD_acc_arr), np.std(QD_acc_arr)
print(SP_acc_mean, SP_acc_std)
print(SA_acc_mean, SA_acc_std)
print(QD_acc_mean, QD_acc_std)