import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegression :
    
    def __init__(self , x , y):
        self.x = x
        self.y = y
    
    def compute_hypothesis(self , theta_0 , theta_1 , x):
        h = theta_0 + np.dot(x , theta_1)
        
        
        return h
    
    def compute_cost(self , theta_0 , theta_1 , x , y):
        h = self.compute_hypothesis(theta_0 , theta_1, x)
        error = 1/(2 * len(x)) * np.sum((h - y) **2)
        return error
    
    def compute_gradients(self , theta_0 , theta_1 , x , y):
        h = self.compute_hypothesis(theta_0 , theta_1, x)
        error = h - y
        d_theta_0 = (1/len(x)) * np.sum(error)
        d_theta_1 = (1/len(x)) * np.dot(error , x)
        
        return d_theta_0 , d_theta_1
    
    def fit(self , learning_rate = 10e-2 , nbr_iterations = 50 , epsilon = 10e-3):
        
        theta_0 , theta_1 = np.random.randn(self.x.shape[0]) , np.random.randn(1)
        costs = []
        
        for _ in range(nbr_iterations):
            
            d_theta_0 , d_theta_1  = self.compute_gradients(theta_0 , theta_1, self.x, self.y)
            
            theta_0 -= learning_rate * d_theta_0
            theta_1 -= learning_rate * d_theta_1
            cost = self.compute_cost(theta_0 , theta_1, self.x, self.y)
            costs.append(cost)
            
            if cost < epsilon :
                break
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.costs = costs
        self.nbr_iterations = nbr_iterations                
    
    def plot_line(self):
        plt.figure()
        plt.scatter(self.x, self.y, color = "blue")
        h = self.compute_hypothesis(self.theta_0 ,self.theta_1, self.x)
        print('h.shape : ',h.shape)
        plt.plot(self.x , h , "-r" ,label="Regression Line")
        plt.xlabel("independent variable")
        plt.ylabel("dependent variable")
        plt.title("Linear Regression Using Gradient Descent")
        plt.legend(loc = 'lower right')
        plt.show()
    
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(1, self.nbr_iterations+1), self.costs, label = r'$J(\theta)$')
        plt.xlabel('Iterations')
        plt.ylabel(r'$J(\theta)$')
        plt.title('Cost vs Iterations of The Gradient Descent')
        plt.legend(loc = 'lower right')

if __name__ == "__main__" :
    x , y = make_regression(n_samples=50 , n_features=1 , noise=10)
    Linear_Regression = LinearRegression(x, y)
    Linear_Regression.fit()
    Linear_Regression.plot_line()
    Linear_Regression.plot_cost()