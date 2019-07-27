import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    """一元线性回归"""
    
    def __init__(self, *args, **kwargs):
        self.max_iter = 1000
        self.lr = 1e-2
        self.w = 0        
        self.b = 0
        
    def loss(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        error = np.sum((self.w * X + self.b - Y) ** 2)
        return error / len(X)
        
    def fit(self, X, Y):
        N = len(X)

        for i in range(self.max_iter):
            g_w = 0; g_b = 0
            for x,y in zip(X,Y):
                g_w += - (2/N) * (y - (self.w * x + self.b)) * x
                g_b += - (2/N) * (y - (self.w * x + self.b))
            self.w -= self.lr * g_w
            self.b -= self.lr * g_b
            if i % 100 == 0:
                print(i,self.loss(X,Y))

    def predict(self, X_test):
        return X_test * self.w + self.b

    def RMSE(y_pred, y_test):
        return np.sqrt(sum(np.square(y_pred - y_test)) / len(y_pred))


if __name__ == '__main__':
    x = [1,2,3,4,5] 
    y =  [1.1, 2.5, 3.3, 4.2, 5]
    
    model = LinearRegression()
    model.fit(x,y)

    plt.scatter(x,y)
    x_fit = np.linspace(1,5)
    y_fit = model.predict(x_fit)
    plt.plot(x_fit, y_fit, color = 'r')
    plt.show()
        