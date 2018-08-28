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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

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


## Basic Variance
def learning_curve(X, y, Xval, yval, _lambda):
    m_train = X.shape[0]
    m_val = Xval.shape[0]
    error_train = np.zeros(m_train)
    error_val = np.zeros(m_train)
    for i in range(m_train):
        data_set = X[0:i+1, :]
        labels = y[0:i+1, :]
        t = trainLinearReg(data_set, labels, _lambda)
        theta_trained = t.x.reshape(X.shape[1], 1)
        h_train = np.dot(data_set, theta_trained)
        h_val = np.dot(Xval, theta_trained)
        error_train[i] = np.sum(np.square(h_train - labels)) / (2*(i+1))
        error_val[i] = np.sum(np.square(h_val - yval)) / (2*m_val)
    return (error_train, error_val)

error_train, error_val = learning_curve(X, y, Xval, yval, 1)
a = np.arange(1, error_train.size + 1)
plt.plot(a, error_train, '-b', label = 'Train')
plt.plot(a, error_val, '-g', label = 'Cross Validation')
plt.axis([0, 13, 0, 150])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.title('Learniing curve for linear regression')
plt.show()

# Polynomial Feature
poly = PolynomialFeatures(degree=8)
X_train_poly = poly.fit_transform(X[:,1].reshape(-1,1))
regr = LinearRegression()
regr.fit(X_train_poly, y)

regr_ply = Ridge(alpha=20)
regr_ply.fit(X_train_poly, y)

plot_x = np.linspace(-60,45)

plot_y = regr.intercept_ + np.sum(regr.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)
plot_y2 = regr_ply.intercept_ + np.sum(regr_ply.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)

plt.plot(plot_x, plot_y, label='Scikit-learn LinearRegression')
plt.plot(plot_x, plot_y2, label='Scikit-learn Ridge (alpha={})'.format(regr_ply.alpha))
plt.scatter(X[:,1], y, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression degree 8')
plt.legend(loc=4)