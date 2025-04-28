# Date: 2018-08-17 8:47
# Author: Enneng Yang
# Abstract：FOBOS

import sys
import matplotlib.pyplot as plt
import random
import numpy as np

# logistic regression
class LR(object):

    @staticmethod
    def fn(w, x):
        ''' sigmod function '''
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        '''cross-entropy loss function'''
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1-y)*np.log(1-y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        '''gradient function'''
        return (y_hat - y) * x

    # 获取多个样本的结果
    @staticmethod
    def fn_multi_samples(w, x):
        ''' sigmod function '''
        return 1.0 / (1.0 + np.exp(-x.dot(w)))

class FTRL(object):

    def __init__(self, dim, l1, l2, alpha, beta, decisionFunc=LR):
        self.dim = dim
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.decisionFunc = decisionFunc

        self.z = np.zeros(dim)
        self.q = np.zeros(dim)
        self.w = np.zeros(dim)

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)


    def update(self, x, y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1
                             else (np.sign(self.z[i]) * self.l1 - self.z[i]) / (self.l2 + (self.beta + np.sqrt(self.q[i]))/self.alpha)
                             for i in range(self.dim)])

        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.q + g*g) - np.sqrt(self.q)) / self.alpha
        self.z += g - sigma * self.w
        self.q += g * g
        return self.decisionFunc.loss(y,y_hat)

    def fit(self, x, y):
        p = self.predict(x)
        self.update(x, y)
        return p


if __name__ ==  '__main__':

   pass


