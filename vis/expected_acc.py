import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def f_theta(theta):
    mean, std = 0, 1
    return norm.pdf(theta, loc=mean, scale=std)

def f_delta(theta, p):
    mean, std = -3, 1
    return norm.pdf((theta + np.log((1-p)/p)), loc=mean, scale=std)

def inner_integrand(theta, p):
    return f_theta(theta) * f_delta(theta, p)

def f_p(p):
    integral, err = quad(inner_integrand, -np.inf, np.inf, args=(p))
    return (1/(p*(1-p))) * integral

def integrand1(p):
    return (1-p)*f_p(p)

def integrand2(p):
    return p*f_p(p)

integral1, err1 = quad(integrand1, 0, 0.5)
integral2, err2 = quad(integrand2, 0.5, 1)
print(integral1)
print(integral2)

S = integral1 + integral2
print(S)

# 0,1; 0,1
# 0,1; 0,3
# 0,1; -3,1
