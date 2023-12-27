import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calc_acc(data_ts: torch.Tensor, predictions: torch.Tensor):
    acc = torch.sum(torch.eq(data_ts, predictions)) / torch.numel(data_ts)
    acc = float(acc)*100
    return round(acc, 3)

def calc_conf_matrix(data_ts: torch.Tensor, predictions: torch.Tensor):
    '''
    i-th row and j-th column entry  indicates the number of samples with true label
    being i-th class and predicted label being j-th class
              Predicted
              0       1 
    True  0|  T0      F1
          1|  F0      T1
    '''
    conf_matrix = confusion_matrix(data_ts.numpy(), predictions.detach().numpy())
    print(conf_matrix)
    conf_matrix = conf_matrix*100/torch.numel(data_ts)
    return conf_matrix.round(3).tolist()

def precision_recall(conf_matrix):
    T0, F1 = conf_matrix[0][0], conf_matrix[0][1]
    F0, T1 = conf_matrix[1][0], conf_matrix[1][1]
    precision = T1/(T1 + F1)
    recall = T1/(T1 + F0)
    return precision*100, recall*100

def calc_RMSE(true, predictions):
    error = true - predictions
    print(torch.sum(torch.abs(error))/torch.numel(error))
    return torch.sqrt(torch.sum(torch.square(error))/torch.numel(error))

def calc_q_acc(data_ts, predictions, q_id_ts):
    acc_dict = {}
    unique_q = torch.unique(q_id_ts) # return tensor of unique question id
    correctness = torch.eq(data_ts, predictions) # return correctness of each entry
    for i in unique_q:
        index_i = torch.where(q_id_ts == i)[0] # index of data entry concerning question i
        correctness_i = torch.index_select(correctness, 0, index_i) # return tensor of data entries concerning question i
        acc_i = torch.sum(correctness_i) / torch.numel(correctness_i)
        acc_dict[i.item()] = round(float(acc_i)*100, 3)
    return acc_dict

def calc_s_acc(data_ts, predictions, s_id_ts):
    acc_dict = {}
    unique_s = torch.unique(s_id_ts) # return tensor of unique question id
    correctness = torch.eq(data_ts, predictions) # return correctness of each entry
    for i in unique_s:
        index_i = torch.where(s_id_ts == i)[0] # index of data entry concerning question i
        correctness_i = torch.index_select(correctness, 0, index_i) # return tensor of data entries concerning question i
        acc_i = torch.sum(correctness_i) / torch.numel(correctness_i)
        acc_dict[i.item()] = round(float(acc_i)*100, 3)
    return acc_dict