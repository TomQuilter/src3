from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def plot_ksdensity(data):
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 100)
    p = kde(x)
    plt.plot(x, p)
