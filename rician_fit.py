import numpy as np
import scipy as sp

from scipy.optimize import fsolve
from scipy.special import iv

def equations(p, measured_mean, measured_variance):
    A, sigma = p
    nu = A**2/ (4.0 * sigma**2.0)
    b =  (1+2.0*nu)*iv(0, nu) + 2.0*nu*iv(1,nu)
    mean = sigma *np.sqrt(np.pi/2.0)*np.exp(-nu)*(b) - measured_mean
    var = A + 2.0*sigma**2.0 - np.pi*sigma**2.0/2.0*np.exp(-2.0*nu)*( b )**2.0 - measured_variance
    return (mean, var)

x, y =  fsolve(equations, (1, 1), (5.0, 3.0))

print(equations((x, y), 5.0, 3.0))
