import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
    
class LogisticRegression(object):
    def __init__(self,lr=0.01, lam=0.1, fit_intercept=True):
        self.lr = lr
        self.lam = lam
        self.max_iter = 1e6
        self.tol = 1e-7         
        self.fit_intercept = fit_intercept

    def _sigmoid(self,x):
        z = 1.0 / (1 + np.exp(-x))
        return z

    def _loss(self,X,y):
        """
        Penalized negative log likelihood of the targets under the current
        model.
            NLL = -1/N * (
                [sum_{i=0}^N y_i log(y_pred_i) + (1-y_i) log(1-y_pred_i)] -
                (gamma ||b||) / 2
            )
        """
        N, M = X.shape
        p = self._sigmoid(np.dot(X, self.w))
        c1 = y * np.log(p)
        c2 = (1 - y) * np.log(1-p)
        loss = (-sum(c1 + c2) + 0.5 * self.lam * sum(self.w ** 2)) / N
        return loss
        # loss = - np.log(p[y==1]).sum() - np.log(1- p[y==0]).sum()
        # penalty = 0.5 * self.lam * np.linalg.norm(self.w, ord=2)**2
        # return (loss + penalty) / N

    def fit(self, X, Y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]   # add bias

        l_prev = np.inf
        self.w = np.random.randn(X.shape[1])

        for i in range(int(self.max_iter)):
            self.w -= self.lr * self._gradient(X,Y)
            loss = self._loss(X,Y)
            if l_prev - loss < self.tol:
                return
            l_prev = loss
            if i % 1000 == 0:
                print(i,'iteration, loss=',loss)
              
    def _gradient(self,X,y):
        y_pred = self._sigmoid(np.dot(X, self.w))
        g_l2norm = self.lam * self.w
        g = - (np.dot(y - y_pred, X) + g_l2norm) / X.shape[0]
        return g

    def predict(self,X):
        if self.fit_intercept:
            X = np.c_(np.ones(X.shape[0]),X)
        y_pred = self._sigmoid(np.dot(X, self.w))
        np.putmask(y_pred, y_pred >= 0.5, 1.0)
        np.putmask(y_pred, y_pred < 0.5, 0.0)
        return y_pred


if __name__ == "__main__":
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    y = np.array([1 if i == 1 else 0 for i in y])

    model = LogisticRegression(lam=0.01)
    model.fit(X,y)


