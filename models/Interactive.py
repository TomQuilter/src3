import torch
import math
import numpy as np

from models.IterativeModel import IterativeModel
from utils.metric_utils.calc_metric import calc_acc

class Interactive(IterativeModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.dimension = model_params['dimension']

    def save_params(self,epoch, bs, bq, xs, xq):
        torch.save(bs,  f'{epoch}_INT_bs.pt')
        torch.save(bq,  f'{epoch}_INT_bq.pt')
        torch.save(xs,  f'{epoch}_INT_xs.pt')
        torch.save(xq,  f'{epoch}_INT_xq.pt')
 
    def train(self, train_ts, val_ts, test_ts, S, Q, rate, iters, init, step_size):
        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        if not init:
            # Randomly initialise random student, question parameters
            bs = torch.randn(S, requires_grad=True, generator=self.rng, dtype=torch.float32) # default
            bq = torch.randn(Q, requires_grad=True, generator=self.rng, dtype=torch.float32) # default
            xs = torch.normal(mean=0, std=np.sqrt(0.1), size=(S, self.dimension), requires_grad=True, generator=self.rng) # default
            xq = torch.normal(mean=0, std=np.sqrt(0.1), size=(Q, self.dimension), requires_grad=True, generator=self.rng) # default
            # bs, bq, xs, xq = torch.load('1975_INT_bs.pt'), torch.load('1975_INT_bq.pt'), torch.load('1975_INT_xs.pt'), torch.load('1975_INT_xq.pt')
          
        else:    
            bs, bq = init['bs'], init['bq']
            


        # xs = torch.ones(size=(S, self.dimension)) * 1
        # xs.requires_grad = True
        # xq = torch.zeros(size=(Q, self.dimension), requires_grad=True)

        last_epoch = iters
        prev_val = 0

        for epoch in range(iters):
            params = {'bs': bs, 'bq': bq, 'xs': xs, 'xq': xq}
            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:   
                val_nll = self.calc_nll(val_ts, params)
                test_nll = self.calc_nll(test_ts, params)

                if epoch != 0 and val_nll > prev_val:
                    last_epoch = epoch
                    #self.save_params(epoch, bs, bq, xs, xq) 
                    # break 
                
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
                xs -= rate * xs.grad
                xq -= rate * xq.grad
 
            # Store bs and xs values
            if epoch == 1600: # (iters-25):
                self.save_params(epoch, bs, bq, xs, xq)
 
            # Zero gradients after updating
            bs.grad.zero_()
            bq.grad.zero_()
            xs.grad.zero_()
            xq.grad.zero_()

            train_nll_arr[epoch] = train_nll
            prev_val = val_nll

        history = {'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1],
                    'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                    'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1],
                    'train acc': np.trim_zeros(train_acc_arr, 'b'),
                    'val acc': np.trim_zeros(val_acc_arr, 'b'),
                    'test acc': np.trim_zeros(test_acc_arr, 'b')}
        params = {'bs': bs, 'bq': bq, 'xs': xs, 'xq': xq}
        return params, history, last_epoch


    def calc_probit(self, data_ts, params):
        bs_data = torch.index_select(params['bs'], 0, data_ts[1])
        bq_data = torch.index_select(params['bq'], 0, data_ts[2])
        xs_data = torch.index_select(params['xs'], 0, data_ts[1])
        xq_data = torch.index_select(params['xq'], 0, data_ts[2])

        # xq_data = torch.exp(xq_data)
        
        interactive_term = torch.sum(xs_data * xq_data, 1) # dot product between xs and xq
        probit_correct = torch.sigmoid(bs_data + bq_data + interactive_term)
        return probit_correct


    def calc_nll(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        nll = -torch.sum(torch.log(probit_correct**data_ts[0]) + torch.log((1-probit_correct)**(1-data_ts[0])))
        return nll


    def predict(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions
