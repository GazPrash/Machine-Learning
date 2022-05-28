# Gradient Descent can be formed with the help of MSE (Cost Function)
# along with coef_ like b & m for single variate linear
# regression models.

# MSE (Mean Squared Error) is a cost function and can be defined as
# mse = summation((yi - yp)**2)/n

# a 3 Dimensional gradient decent can be formed when we plot
# MSE, b & m together, here b == intercept & m = ceof_ of one of
# the possible linear regression lines, the combination
# of b & m can be obtained by minimizing the MSE output, 
# which on the 3D surface will be the lowest point (or the global
# minima) of the descent.

# funcMSE = 1/n * sum((yi - (mxi + b))**2), start = 1, fin = n)
# then funcMSE/delta(m) & funcMSE/delta(b) can be calculated using
# partial differentiation

# for more info or rev use : 
#     https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=4

import numpy as np
import random as rd
import matplotlib.pyplot as plt


def gradient_descent(x:np.array, y:np.array, iterations:int)->tuple:
    if (len(x) != len(y)): return -1    
    n = len(x)

    coef_ = 0
    intercept = 0
    learning_rate = 10e-3

    for i in range(iterations):
        y_pred = coef_ * x + intercept
        mse_costfunc = (1/n) * sum([j*j for j in (y-y_pred)])
        m_deriv = -(2/n) * sum(x*(y - y_pred))
        b_deriv = -(2/n) * sum(y - y_pred)
        coef_ -= (learning_rate * m_deriv)
        intercept -= (learning_rate * b_deriv)

        print(f"COST : {mse_costfunc} | ITERATION : {i}")

    return ((coef_, intercept))

x = np.array([4, 5, 2, 6, 7])
y = np.array([3, 5, 11, 5, 8])

m, b = gradient_descent(x, y, 10000)

print(f"Coef_ is : {m} & Intercept is : {b}")



