import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

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
    
    def compute_cost(self , X , theta):
        hypothesis = self.compute_hypothesis(X, theta)
        n_samples = len(self.y)
        error = hypothesis - self.y
        cost = (1/2 * n_samples) * np.sum((error ) ** 2 )
        return cost
    
    def Standardize_Data(self ,x):
        return (x - np.mean(x)) / (np.max(x) - np.min(x))
    
    def fit(self , order = 2 , epsilon = 10e-3 , nbr_iterations = 1000 , learning_rate = 10e-1):
        
        self.order = order
        self.nbr_iterations = nbr_iterations
        
        #X = [self.x ** i for i in range(order+1)]
        X = []
        X.append(np.ones(len(self.x)))
        for i in range(1 , order + 1):
            X.append(self.Standardize_Data(self.x ** i))
        
        X = np.column_stack(X)
        theta = np.random.randn(order+1)
        costs = []
        
        for i in range(self.nbr_iterations):
            
            # Computing The Hypothesis for the current params (theta)
            hypothesis = self.compute_hypothesis(X, theta)
            
            # Computing The Errors
            errors =  hypothesis - self.y
            
            # Update Theta Using Gradient Descent 
            n_samples = len(self.y)
            d_J = (1/ n_samples) * np.dot(X.T , errors)
            theta -= learning_rate *  d_J 
            
            # Computing The Cost
            cost = self.compute_cost(X, theta)
            costs.append(cost)            
            
            # if the current cost less than epsilon stop the gradient Descent
            
            if cost < epsilon :
                break
            
        self.costs = costs
        self.X = X
        self.theta = theta    
        
    def plot_line(self):
        plt.figure()
        plt.scatter(self.x , self.y , color = 'blue')
        
        # Line for Order 1
        Y_hat = self.compute_hypothesis(self.X , self.theta)
        plt.plot(self.x , Y_hat , "-r" , label = 'Order = ' + str(self.order) )
        
        # Line for Order 2
        self.fit(order = 2)
        Y_hat = self.compute_hypothesis(self.X , self.theta)
        plt.plot(self.x , Y_hat , "-g" , label = 'Order = ' + str(self.order) )
        
        # Line for Order 3
        self.fit(order = 3)
        Y_hat = self.compute_hypothesis(self.X , self.theta)
        plt.plot(self.x , Y_hat , "-m" , label = 'Order = ' + str(self.order) )
        
        # Line for Order 4
        self.fit(order = 4)
        Y_hat = self.compute_hypothesis(self.X , self.theta)
        plt.plot(self.x , Y_hat , "-y" , label = 'Order = ' + str(self.order) )
        
        plt.xlabel("independent variable")
        plt.ylabel("dependent variable")
        plt.title("Polynomial Regression Using Gradient Descent")
        plt.legend(loc = 'lower right')
        plt.show()
        
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(1, self.nbr_iterations+1), self.costs, label = r'$J(\theta)$')
        plt.xlabel('Iterations')
        plt.ylabel(r'$J(\theta)$')
        plt.title('Cost vs Iterations of The Gradient Descent')
        plt.legend(loc = 'lower right')
        
if __name__ == "__main__":
    x,y = Generate_Points(0, 50, 100, [3, 1, 1], 2.3)
    Poly_regression = Polynomial_Reression(x, y)
    Poly_regression.fit(order = 1)
    Poly_regression.plot_line()
    Poly_regression.plot_cost()