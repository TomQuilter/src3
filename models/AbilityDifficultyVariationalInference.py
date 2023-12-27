import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

torch.manual_seed(0)

from models.IterativeModel import IterativeModel
from utils.metric_utils.calc_metric import calc_acc
 
class AbilityDifficultyVariationalInference(IterativeModel):
    def __init__(self, model_params):
        super().__init__(model_params)

    def train(self, train_ts, val_ts, test_ts, S, Q, rate, iters, init, step_size):
        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        # Randomly ### initialise random student ####, question parameters
        # bs = torch.zeros(S, requires_grad=True)
        # bq = torch.zeros(Q, requires_grad=True)


        #bs = torch.randn(S, requires_grad=True, generator=self.rng)  ## Creates a Bs for each student
        #print("bs1", bs)
        #print("hi1")
        ## initialise Mu and sigma for each student
        Ms = torch.zeros(S, requires_grad=True)
        Ss = torch.ones(S, requires_grad=True)    
         
        # Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        epsilon = torch.randn(S)  # samples from a standard normal (mean=0, std=1)   S = Number of STUDENTS
        bs = Ms + 0.1*Ss * epsilon  # reparameterization trick
        #p# print("bs2", bs)

        bq = torch.randn(Q, requires_grad=True, generator=self.rng)

        last_epoch = iters
        prev_nll = 0
        for epoch in range(iters):
            params = {'Ms': Ms, 'Ss': Ss, 'bq': bq} # {'bs': bs, 'bq': bq}
            #print("hi2") 
            train_nll = self.calc_nll(train_ts, params)
            #print("hibeforebackward")
            train_nll.backward()             ## This is taking the time!!! The Grad descent
            #print("hiafterbackward")
            if epoch % step_size == 0:
                val_nll = self.calc_nll(val_ts, params)
                test_nll = self.calc_nll(test_ts, params)

                # terminate at last iteration if test nll of this iter greater than test nll of last iter
                if epoch != 0 and 0.9*test_nll > prev_nll:
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
                # bs -= rate * bs.grad  ## All Bs's updated, needs to be Mu and sigma now
                Ms -= rate * Ms.grad
                Ss -= rate * Ss.grad
                bq -= rate * bq.grad
 
            # Zero gradients after updating
            #bs.grad.zero_()
            Ms.grad.zero_()
            Ss.grad.zero_()
            bq.grad.zero_()
 
            train_nll_arr[epoch] = train_nll
            prev_nll = test_nll

        history = {'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1], 
                    'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                    'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1], 
                    'train acc': np.trim_zeros(train_acc_arr, 'b'),
                    'val acc': np.trim_zeros(val_acc_arr, 'b'),
                    'test acc': np.trim_zeros(test_acc_arr, 'b')}    ### Bs = mu + here to change params to Ms and SS?
        params = {'Ms': Ms, 'Ss': Ss, 'bq': bq} # {'bs': bs, 'bq': bq}
        return params, history, last_epoch
    
#    # KL Divergence between two normal distributions
#    def kl_divergence_normal(self, mean1, var1, mean2, var2):
#       return np.log(np.sqrt(var2) / np.sqrt(var1)) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5

    def kl_divergence_normal(self, mean1, var1, mean2, var2):
    # Ensure that the input tensors are all floats for the division and sqrt operations
        mean1, var1, mean2, var2 = map(torch.as_tensor, (mean1, var1, mean2, var2))
        return torch.log(torch.sqrt(var2) / torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5

    def calc_probit(self, data_ts, params):
        #p#print("params['Ms']",params['Ms'])
        #p#print("params['Ss']",params['Ss']) 
        NumberofStudents = len(params['Ss'])
        #p# print("NumberofStudents",NumberofStudents)
        #Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        epsilon = torch.randn(NumberofStudents)  # samples from a standard normal (mean=0, std=1)
        params['bs'] = params['Ms'] + 0.1*params['Ss'] * epsilon  # reparameterization trick
        #print("bs2", bs)
        #p#print("params['bs']")
        #p#print(params['bs'])

        bs_data = torch.index_select(params['bs'], 0, data_ts[1])  ### Grab params['bs'] , grad MuS and SS and them generate no!?  
        #p#print("bs_data") 
        #p#print(bs_data) 
        ## EVERY time u want Bs (for all students) u take a draw and get a 1 by S vector
        ## So replace Bs with Mu and Sigma and generate the Bs's
        ## select with data_ts[1]) an issue here?? No just takes a certain bs value - bs is just a 1 by S list of numbers
        bq_data = torch.index_select(params['bq'], 0, data_ts[2])
 
        probit_correct = torch.sigmoid(bs_data + bq_data)

        return probit_correct, params
 
    def calc_nll(self, data_ts, params):
        ## Loop over M samples of Bs with Ms and Ss and work out a Mean M nll 
        ## For M = 1 to M for x in range(6):
        #print("params['Ms']",params['Ms'])
        #print("params['Ss']",params['Ss']) 
        nllsum = 0
        MStudentsToSample = 100  
        for SampleNumber in range(MStudentsToSample):
            #print("hi3") 
            probit_correct, params = self.calc_probit(data_ts, params)
            kl_divergences = []
            kl_divergences = self.kl_divergence_normal(params['Ms'], params['Ss'], torch.zeros_like(params['Ms']), torch.ones_like(params['Ss']))
            Total_kl_divergence = torch.sum(kl_divergences)
            #for mean, var in zip(params['Ms'], params['Ss']):    ### Does this for each student!!! A big for loop hmmmmmm THIS IS SLOWING IT DOWN
            #    #print("hi4")
            #    kl_div = self.kl_divergence_normal(mean.item(), var.item(), 0, 1)
            #    kl_divergences.append(kl_div)
            #   
            #Total_kl_divergence = np.sum(kl_divergences)  ## ForThisSetOfSampledStudentsMeanandVariances   ## Doing this in Batches
                #print("hi5")
            nll = -torch.sum(data_ts[0]*torch.log(probit_correct) + (1-data_ts[0])*torch.log(1-probit_correct)) + Total_kl_divergence
            ## KL is a punishment for how far the distributions are from a standard normal, remember bs is around 0
            if nll < 0 : 
                1==1
            nllsum += nll 
        nllaverage = nllsum/MStudentsToSample 
        ## Sum the nlls and average at the end
        return nllaverage
  

    def predict(self, data_ts, params):
        probit_correct, params = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions
