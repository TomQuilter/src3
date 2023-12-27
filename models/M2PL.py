import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn.functional import normalize

from models.IterativeModel import IterativeModel
from utils.metric_utils.calc_metric import calc_acc

class M2PL(IterativeModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.dimension = model_params['dimension']


    def train(self, train_ts, val_ts, test_ts, S, Q, rate, iters, init, step_size):
        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        # Randomly initialise random student, question parameters
        bs = torch.normal(mean=0, std=np.sqrt(0.1), size=(S, self.dimension+1), requires_grad=True, generator=self.rng)
        bq = torch.normal(mean=0, std=np.sqrt(0.1), size=(Q, self.dimension+1), requires_grad=True, generator=self.rng)

        last_epoch = iters
        prev_val = 0

        for epoch in range(iters):
            params = {'bs': bs, 'bq': bq}
            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:
                print(epoch, bs[:10])
                print(epoch, bq[:10])
            
                val_nll = self.calc_nll(val_ts, params)
                test_nll = self.calc_nll(test_ts, params)

                if epoch != 0 and val_nll > prev_val:
                    last_epoch = epoch
                    break
                
                val_nll_arr[epoch//step_size] = val_nll
                test_nll_arr[epoch//step_size] = test_nll
                
                train_acc = calc_acc(train_ts[0], self.predict(train_ts, params)[1])
                val_acc = calc_acc(val_ts[0], self.predict(val_ts, params)[1])
                test_acc = calc_acc(test_ts[0], self.predict(test_ts, params)[1])
                train_acc_arr[epoch//step_size], val_acc_arr[epoch//step_size], test_acc_arr[epoch//step_size] = train_acc, val_acc, test_acc

                self.print_iter_res(epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc)

            # Gradient descent
            with torch.no_grad():
                bs -= rate * bs.grad
                bq -= rate * bq.grad

            # Zero gradients after updating
            bs.grad.zero_()
            bq.grad.zero_()

            train_nll_arr[epoch] = train_nll
            prev_val = val_nll

        history = {'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1],
                    'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                    'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1],
                    'train acc': np.trim_zeros(train_acc_arr, 'b'),
                    'val acc': np.trim_zeros(val_acc_arr, 'b'),
                    'test acc': np.trim_zeros(test_acc_arr, 'b')}
        params = {'bs': bs, 'bq': bq}
        return params, history, last_epoch


    def calc_probit(self, data_ts, params):
        bs_data = torch.index_select(params['bs'], 0, data_ts[1])
        bq_data = torch.index_select(params['bq'], 0, data_ts[2])
        
        bs0_data = bs_data[:, 0]
        bq0_data = bq_data[:, 0]

        if self.dimension == 0:
            probit_correct = torch.sigmoid(bs0_data + bq0_data)
        else:
            xs_data = bs_data[:, 1:]
            xq_data = bq_data[:, 1:]
            interactive_term = torch.sum(xs_data * xq_data, 1) # dot product between xs and xq
            probit_correct = torch.sigmoid(bs0_data + bq0_data + interactive_term)

        return probit_correct


    def calc_nll(self, data_ts, params, l=0):
        probit_correct = self.calc_probit(data_ts, params)
        nll = -torch.sum(torch.log(probit_correct**data_ts[0]) + torch.log((1-probit_correct)**(1-data_ts[0])))
        
        if self.dimension > 0:
            # Regularise bq
            # xs = params['bs'][:, 1:]
            xq = params['bq'][:, 1:]
            xq_norm = torch.norm(xq, dim=1)
            penalty = torch.sum(torch.square(xq_norm))
            nll += (l * penalty)
        
        return nll


    def predict(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions


    def synthesise(self, synthetic_params, save=False):
        dimension = synthetic_params['dimension']
        S, Q = synthetic_params['student_dimension'], synthetic_params['question_dimension']
        self.rng.manual_seed(synthetic_params['seed'])
        bs_mean, bs_std = synthetic_params['bs_mean'], synthetic_params['bs_std']
        bq_mean, bq_std = synthetic_params['bq_mean'], synthetic_params['bq_std']

        bs = torch.normal(mean=bs_mean, std=bs_std, size=(S, 1), requires_grad=True, generator=self.rng)
        bq = torch.normal(mean=bq_mean, std=bq_std, size=(Q, 1), requires_grad=True, generator=self.rng)

        bs0 = bs[:, 0]
        bq0 = bq[:, 0]
        bs0_matrix = bs0.repeat(Q, 1).T
        bq0_matrix = bq0.repeat(S, 1)

        if dimension == 0:
            probit_correct = torch.sigmoid(bs0_matrix + bq0_matrix)
        else:
            xs_mean, xs_std = synthetic_params['xs_mean'], synthetic_params['xs_std']
            xq_mean, xq_std = synthetic_params['xq_mean'], synthetic_params['xq_std']
            xs = torch.normal(mean=xs_mean, std=xs_std, size=(S, dimension), requires_grad=True, generator=self.rng)
            xq = torch.normal(mean=xq_mean, std=xq_std, size=(Q, dimension), requires_grad=True, generator=self.rng)
            interactive_matrix = torch.matmul(xs, xq.T)
            probit_correct = torch.sigmoid(bs0_matrix + bq0_matrix + interactive_matrix)
            bs = torch.concat([bs, xs], dim=1)
            bq = torch.concat([bq, xq], dim=1)
        
        self.rng.manual_seed(synthetic_params['seed']+1000)
        synthetic_data = torch.bernoulli(probit_correct, generator=self.rng)
        synthetic_df = pd.DataFrame(synthetic_data).astype(float)
        ground_truth_params = {'bs': bs, 'bq': bq}

        if save:
            torch.save({'synthetic_df': synthetic_df, 'ground_truth_params': ground_truth_params}, save)

        return synthetic_df, ground_truth_params
