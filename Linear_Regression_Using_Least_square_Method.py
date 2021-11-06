import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x , y = make_regression(n_samples=100 , n_features= 1 , noise=10)

"""
Our Linear Model Looks like this :   Y_estimated = theta[0] + theta[1] * x .
our goal is to compute the coefficients Theta[0] and Theta[1] that will fit our data correctly,
Using Least Square Method . 

"""

# Calculating The Mean for x and y
X_mean = np.mean(x)
Y_mean = np.mean(y)

# Calculating The Coefficient Theta_1
num = np.sum( (x[i] - X_mean)*(y[i] - Y_mean) for i in range(len(x)) )
den = np.sum( (x[i] - X_mean)**2 for i in range(len(x)) )
theta_1 = num / den

# Calculating The Coefficient Theta_0
theta_0 = Y_mean - theta_1 * X_mean

# Calculating The Estimated Y
Y_estimated = theta_0 + theta_1 * x

# Plot The Result

plt.figure()
plt.scatter(x , y , color='blue')
plt.plot(x , Y_estimated ,"-r" ,label = 'Estimated Value' )
plt.xlabel('Independent Variable')
plt.xlabel('dependent Variable')
plt.title('Linear Regression Using Least Square Method')
plt.legend(loc = 'lower right')
plt.show()
