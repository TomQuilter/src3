import torch
import numpy as np

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for training.")
else:
    print("CUDA is not available. Training will use CPU.")

def printevennumbers

# Define the parameters for the distributions
params = {
    'Ms': torch.tensor([0., 0., 0.]),  # Means
    'Ss': torch.tensor([0.5, 1., 1.])   # Variances
}

# KL Divergence between two normal distributions
def kl_divergence_normal(mean1, var1, mean2, var2):
    return np.log(np.sqrt(var2) / np.sqrt(var1)) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5

# Calculate KL Divergence for each pair and average them
kl_divergences = []
for mean, var in zip(params['Ms'], params['Ss']):
    kl_div = kl_divergence_normal(mean.item(), var.item(), 0, 1)
    kl_divergences.append(kl_div)

average_kl_divergence = np.mean(kl_divergences)
print("Average KL Divergence:", average_kl_divergence)
