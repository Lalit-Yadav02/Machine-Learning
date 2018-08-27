#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:47:06 2018

@author: lality
"""

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.optimize import minimize

plt.rcParams['figure.figsize'] = (10.0, 7.0)
np.set_printoptions(suppress=True)

# Loading data
data = loadmat('data/ex5data1.mat')
X = data['X']
y = data['y']

# Function for plotiing data
def plot_data(X, y, xlabel, ylabel, xmin, xmax, ymin, ymax, axes = None):
    plt.plot(X, y, 'rx')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

plot_data(X, y, 'Change in water level(x)', 'Water flowing out of the dam(y)', -50, 40, 0, 40)

# Function to compute linear regression cost
def linearRegCostFunction(theta, X, y, _lambda):
    m = len(y)
    J = 0
    grad = np.zeros(theta.size)
    theta = theta.reshape(X.shape[1], 1)
    h = np.dot(X, theta)
    J = (1/(2*m)) * np.dot((h - y).T, h - y) + (_lambda/(2 * m)) * np.sum(np.square(theta[1:]))
    grad = ((1/m) * (h - y).T.dot(X)).T + (_lambda/m) * np.r_[[[0]], theta[1:]]
    grad = np.r_[grad.ravel()]
    return (J, grad)

X = np.c_[np.ones(X.shape[0]), X]

Xval = np.c_[np.ones(data['Xval'].shape[0]), data['Xval']]
yval = data['yval']

Xtest = np.c_[np.ones(data['Xtest'].shape[0]), data['Xtest']]
ytest = data['yval']

theta = np.array([np.ones([2])]).T
print(X.shape, '*', theta.shape, '=?', y.shape)
print('J = ', linearRegCostFunction(theta, X, y, 1)[0][0][0])
print('grad = ', linearRegCostFunction(theta, X, y, 1)[1])

# Train Linear Regression
def trainLinearReg(X, y, _lambda):
    initial_theta = np.zeros([X.shape[1], 1])
    theta = minimize(fun = linearRegCostFunction, x0 = initial_theta, args = (X, y, _lambda), method = 'CG', jac=True, options = {'maxiter': 200})
    return theta

trained = trainLinearReg(X, y, 0)
print(trained.x[0], trained.x[1])

plot_data(X, y, 'Change in water level(x)', 'Water flowing out of the dam(y)', -50, 40, -5, 40)

plt.plot([-50, 40], [-1*trained.x[1]*50 + trained.x[0], trained.x[1]*40 + trained.x[0]])
########################

def trainLinearReg(X, y, lmda):
    initial_theta = np.zeros( [X.shape[1], 1] )
    theta = minimize( fun = linearRegCostFunction, x0 = initial_theta, 
              args = (X, y, lmda), 
              method = 'CG', jac = True, options = {'maxiter' : 200} )
    return theta

trained = trainLinearReg(X, y, 0)
print(trained.x[0], trained.x[1])

plot_dataset( data1['X'], data1['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -50, 40, -5, 40 )
plt.plot([-50, 40], [-1*trained.x[1]*50 + trained.x[0] , trained.x[1]*40 + trained.x[0]] )