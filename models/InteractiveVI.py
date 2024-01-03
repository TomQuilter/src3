import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

torch.manual_seed(0)

from models.IterativeModel import IterativeModel
from utils.metric_utils.calc_metric import calc_acc
### Sanity checked with zero variance Dec 4th
class InteractiveVI(IterativeModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.dimension = model_params['dimension']
 

    def train(self, train_ts, val_ts, test_ts, S, Q, rate, iters, init, step_size):
        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        if not init:
            # Randomly initialise random student, question parameters
            # bs = torch.randn(S, requires_grad=True, generator=self.rng, dtype=torch.float32) # default

            ## initialise Mu and sigma for each student
            Ms = torch.zeros(S, requires_grad=True)
            Ss = torch.ones(S, requires_grad=True)    
        
            # Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
            epsilon = torch.randn(S)  # samples from a standard normal (mean=0, std=1)   S = Number of STUDENTS
            bs = Ms + Ss * epsilon  # reparameterization trick
            # bs = torch.randn(S, requires_grad=True, generator=self.rng, dtype=torch.float32) # default
            #p# print("bs2", bs)
            bs = torch.load('975_bs.pt')
 
            bq = torch.randn(Q, requires_grad=True, generator=self.rng, dtype=torch.float32) # default
            bq = torch.load('975_bq.pt')
        else:    
            bs, bq = init['bs'], init['bq']
        
 
        ## So xs will also need VI on them! ie. initiate , grad update
 
        ## initialise Mu and sigma for each student
        xms = torch.zeros(size=(S, self.dimension), requires_grad=True)
        xms = torch.load('975_xs.pt')
        print("initial xms",xms) 
        xss = torch.ones(size=(S, self.dimension), requires_grad=True)    
        # xss = torch.load('975_xs.pt')

        # Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        ## Need a D dimensional vector 
        epsilon = torch.randn(S, self.dimension)  # samples from a standard normal (mean=0, std=1)   S = Number of STUDENTS
        xs = xms + xss * epsilon  # reparameterization trick
        print("xs",xs)
        #xs = torch.randn(S, requires_grad=True, generator=self.rng, dtype=torch.float32) # default

        # xms = 
        # xss = 
        
        #xs = torch.normal(mean=0, std=np.sqrt(0.1), size=(S, self.dimension), requires_grad=True, generator=self.rng) # default
        ## self.dimension = number of dimensions hey

        xq = torch.normal(mean=0, std=np.sqrt(0.1), size=(Q, self.dimension), requires_grad=True, generator=self.rng) # default

        # xs = torch.ones(size=(S, self.dimension)) * 1
        # xs.requires_grad = True
        # xq = torch.zeros(size=(Q, self.dimension), requires_grad=True)

        last_epoch = iters
        prev_val = 0

        for epoch in range(iters):
            # params = {'Ms': Ms, 'Ss': Ss, 'bq': bq, 'xs': xs, 'xq': xq}      ## params = {'bs': bs, 'bq': bq, 'xs': xs, 'xq': xq} 
            params = {'Ms': Ms, 'Ss': Ss, 'bq': bq, 'xms': xms, 'xss': xss, 'xq': xq}
            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:   
                val_nll = self.calc_nll(val_ts, params)
                test_nll = self.calc_nll(test_ts, params)

                if epoch != 0 and 0.8*val_nll > prev_val:
                    last_epoch = epoch
                    #break 
                
                val_nll_arr[epoch//step_size] = val_nll
                test_nll_arr[epoch//step_size] = test_nll

                
                train_acc = calc_acc(train_ts[0], self.predict(train_ts, params)[1])
                val_acc = calc_acc(val_ts[0], self.predict(val_ts, params)[1])
                test_acc = calc_acc(test_ts[0], self.predict(test_ts, params)[1])
                train_acc_arr[epoch//step_size], val_acc_arr[epoch//step_size], test_acc_arr[epoch//step_size] = train_acc, val_acc, test_acc

                self.print_iter_res(epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc)

            # Gradient descent
            with torch.no_grad():
                #bs -= rate * bs.grad
                Ms -= rate * Ms.grad
                Ss -= rate * Ss.grad
                bq -= rate * bq.grad
                #xs -= rate * xs.grad
                xms -= rate * xms.grad
                xss -= rate * xss.grad
                xq -= rate * xq.grad

            # Zero gradients after updating
            #bs.grad.zero_()
            Ms.grad.zero_()
            Ss.grad.zero_()
            bq.grad.zero_()
            #xs.grad.zero_()
            xms.grad.zero_()
            xss.grad.zero_()
            xq.grad.zero_()
 
            train_nll_arr[epoch] = train_nll
            prev_val = val_nll

        history = {'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1],
                    'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                    'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1],
                    'train acc': np.trim_zeros(train_acc_arr, 'b'),
                    'val acc': np.trim_zeros(val_acc_arr, 'b'),
                    'test acc': np.trim_zeros(test_acc_arr, 'b')}
        #params = {'Ms': Ms, 'Ss': Ss, 'bq': bq, 'xs': xs, 'xq': xq}      ## params = {'bs': bs, 'bq': bq, 'xs': xs, 'xq': xq} 
        params = {'Ms': Ms, 'Ss': Ss, 'bq': bq, 'xms': xms, 'xss': xss, 'xq': xq}
        return params, history, last_epoch

    # KL Divergence between two normal distributions
    def kl_divergence_normal(self, mean1, var1, mean2, var2):
    # Ensure that the input tensors are all floats for the division and sqrt operations
        mean1, var1, mean2, var2 = map(torch.as_tensor, (mean1, var1, mean2, var2))
        return torch.log(torch.sqrt(var2) / torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5
     
    def calc_probit(self, data_ts, params):
        # bs_data = torch.index_select(params['bs'], 0, data_ts[1])
 
        #p#print("params['Ms']",params['Ms'])
        #p#print("params['Ss']",params['Ss']) 
        NumberofStudents = len(params['Ss'])
        #p# print("NumberofStudents",NumberofStudents)
        #Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        epsilon = torch.randn(NumberofStudents)  # samples from a standard normal (mean=0, std=1)
        params['bs'] = params['Ms'] + params['Ss'] * epsilon  # reparameterization trick
        #print("bs2", bs)
        #p#print("params['bs']")
        #p#print(params['bs'])
 
        bs_data = torch.index_select(params['bs'], 0, data_ts[1])  ### Grab params['bs'] , grad MuS and SS and them generate no!?  
        
        #print("bs_data") 
        #print(bs_data) 

        #torch.save(bs_data, 'tensor.pt')

        #x = torch.load('tensor.pt')

        #print("x", x)
 
        #np.savetxt('my_file.txt', torch.Tensor([3,4,5,6]).numpy())
        #np.save('final_bs_values.npy', bs_data)

        ## EVERY time u want Bs (for all students) u take a draw and get a 1 by S vector
        ## So replace Bs with Mu and Sigma and generate the Bs's
        ## select with data_ts[1]) an issue here?? No just takes a certain bs value - bs is just a 1 by S list of numbers

        bq_data = torch.index_select(params['bq'], 0, data_ts[2])

        ### VI the Xs ###

        epsilon = torch.randn(NumberofStudents, self.dimension)  # samples from a standard normal (mean=0, std=1)   S = Number of STUDENTS
        params['xs'] = params['xms'] + params['xss'] * epsilon  # reparameterization trick
 
        xs_data = torch.index_select(params['xs'], 0, data_ts[1])

        xq_data = torch.index_select(params['xq'], 0, data_ts[2])

        # xq_data = torch.exp(xq_data)
        
        interactive_term = torch.sum(xs_data * xq_data, 1) # dot product between xs and xq
        probit_correct = torch.sigmoid(bs_data + bq_data + interactive_term)
        return probit_correct, params

 
    def calc_nll(self, data_ts, params):
        ## Loop over M samples of Bs with Ms and Ss and work out a Mean M nll
        ## For M = 1 to M for x in range(6):
        #print("params['Ms']",params['Ms'])
        #print("params['Ss']",params['Ss'])
        nllsum = 0
        MStudentsToSample = 25 
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
        probit_correct, updated_params = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions
