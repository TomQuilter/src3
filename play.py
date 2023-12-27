import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

# Bs as a sample disturbution 

number_of_samples = 5
Mean = 0
StandDev = 1
Mu_s = 0
sigma_s = 2

samples = np.random.normal(Mean, StandDev, number_of_samples)

print(samples)

#### ReParametrisation TRICK! ####
Bs = Mu_s+sigma_s*samples

print(Bs)

## Mu_s = 0 and sigma_s get grad descent updated

S = 4

#bs = torch.randn(S, requires_grad=True, generator=self.rng)
#print(bs)

import torch

Ms = torch.zeros(S, requires_grad=True)  # means
Ss = torch.ones(S, requires_grad=True)  # standard deviations

# Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
epsilon = torch.randn(S)  # samples from a standard normal (mean=0, std=1)
bs = Ms + Ss * epsilon  # reparameterization

print(bs)




