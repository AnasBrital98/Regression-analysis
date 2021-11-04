import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def Generate_Points(start , end , nbr_points , coefficient , noise ):
    
    #Creating X 
    x = np.arange(start , end , (end -start) / nbr_points)
    
    #calculating Y
    y = coefficient[0]
    for i in range(1 , len(coefficient)) :
        y += coefficient[i] * x ** i
    
    #Adding noise to Y
    if noise != 0 :
        y += np.random.normal(-(10 ** noise) , 10**noise , len(x))
    
    return x,y

"""
You can generate Polynomial Points Using The Function Above , or You can Use 
The Function in Sklearn like this :
    
    from sklearn.datasets import make_regression
    from matplotlib import pyplot
    x, y = make_regression(n_samples=150, n_features=1, noise=0.2)
    pyplot.scatter(x,y)
    pyplot.show()

"""

class Polynomial_Reression :
    
    def __init__(self , x , y ):
        self.x = x
        self.y = y
    
    def compute_hypothesis(self , X , theta):
        hypothesis = np.dot(X , theta)
        return hypothesis
    
    def fit(self , order = 2 , epsilon = 10e-3 , nbr_iterations = 100 , learning_rate = 10e-3):
        
        self.order = order
        self.nbr_iterations = nbr_iterations
        
        X = [self.x ** i for i in range(order+1)]
        X = np.column_stack(X)
        theta = np.random.randn(order+1)
        costs = []
        
        
        
        pass
            
    def plot_line(self):
        plt.figure()
        plt.scatter(self.x , self.y , color = 'blue')
        Y_hat = self.compute_hypothesis(self.X , self.theta)
        plt.plot(self.x , Y_hat , "-r")
        plt.xlabel("independent variable")
        plt.ylabel("dependent variable")
        plt.title("Polynomial Regression Using Gradient Descent")
        plt.show()
        
if __name__ == "__main__":
    x,y = Generate_Points(0, 50, 100, [3,2,1], 1.5)
    Poly_regression = Polynomial_Reression(x, y)
    Poly_regression.fit(order = 3)
    Poly_regression.plot_line()