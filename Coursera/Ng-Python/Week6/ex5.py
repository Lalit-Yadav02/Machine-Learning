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
def poly_features(X, p):
    X_poly = X.reshape(X.shape[0], 1)
    for i in range(1, p):
        X_poly = np.c_[X_poly, np.power(X_poly[:,0], i + 1)]
    return X_poly

def feature_normalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

p = 8
X_poly = poly_features(data['X'], p)

X_norm, mu, sigma = feature_normalize(X_poly)
X_poly = np.c_[np.ones(X_poly.shape[0]), X_norm]

X_poly_test = poly_features(data['Xtest'], p)
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), (X_poly_test - mu)/sigma]

X_poly_val = poly_features(data['Xtest'], p)
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), (X_poly_val - mu)/ sigma]

_lambda = 0.0 
theta = trainLinearReg(X_poly, y, _lambda).x.reshape(X_poly.shape[1], 1)

plot_data(data['X'], data['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -80, 80, -60, 40 )


########################

In [17]:
plot_dataset( data1['X'], data1['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -80, 80, -60, 40 )
x = np.arange( np.amin(data1['X']) - 15, np.amax(data1['X']) + 25, 0.05 )
x_poly = poly_features(x, p)
x_poly = np.c_[ np.ones(x_poly.shape[0]), (x_poly - mu)/sigma ]
plt.title(f'Polynomial Regression Fit (lambda = {lmda})') # python 3.6
plt.plot( x, x_poly.dot(theta), '--' )

error_train, error_val = learning_curve( X_poly, y, X_poly_val, yval, lmda )

plt.plot( np.arange(1, error_train.size + 1), error_train, '-b', label = 'Train' )
plt.plot( np.arange(1, error_train.size + 1), error_val, '-g', label = 'Cross Validation' )
plt.axis([0, error_train.size + 1, 0, 100])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.title(f'Polynomial Regression Learning Curve (lambda = {lmda})')
plt.show()


lmda_1 = 1
theta_1 = trainLinearReg( X_poly, y, lmda_1 ).x.reshape(X_poly.shape[1],1)
In [20]:
plot_dataset( data1['X'], data1['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -80, 80, 0, 160 )
x = np.arange( np.amin(data1['X']) - 15, np.amax(data1['X']) + 25, 0.05 )
x_poly = poly_features(x, p)
x_poly = np.c_[ np.ones(x_poly.shape[0]), (x_poly - mu)/sigma ]
plt.title(f'Polynomial Regression Fit (lambda = {lmda_1})') # python 3.6
plt.plot( x, x_poly.dot(theta_1), '--' )

error_train, error_val = learning_curve( X_poly, y, X_poly_val, yval, lmda_1 )
a = np.arange(1,error_train.size + 1)
plt.plot( a, error_train, '-b', label = 'Train' )
plt.plot( a, error_val, '-g', label = 'Cross Validation' )
plt.axis([0, error_train.size + 1, 0, 100])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.title(f'Polynomial Regression Learning Curve (lambda = {lmda_1})')
plt.show()

lmda_100 = 100
theta_100 = trainLinearReg( X_poly, y, lmda_100 ).x.reshape(X_poly.shape[1],1)
plot_dataset( data1['X'], data1['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -80, 80, -10, 40 )
x = np.arange( np.amin(data1['X']) - 15, np.amax(data1['X']) + 25, 0.05 )
x_poly = poly_features(x, p)
x_poly = np.c_[ np.ones(x_poly.shape[0]), (x_poly - mu)/sigma ]
plt.title(f'Polynomial Regression Fit (lambda = {lmda_100})') # python 3.6
plt.plot( x, x_poly.dot(theta_100), '--' )

def validation_curve( X, y, Xval, yval ):
    lmda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    m_train = X.shape[0]
    m_val = Xval.shape[0]
    error_train = np.zeros(lmda_vec.size)
    error_val   = np.zeros(lmda_vec.size)
    for i in range(lmda_vec.size):
        lmda = lmda_vec[i]
        theta_trained = trainLinearReg( X, y, lmda ).x.reshape(X.shape[1],1)
        h_train = np.dot( X, theta_trained )
        h_val = np.dot( Xval, theta_trained )
        error_train[i] = np.sum(np.square( h_train - y ))/(2*(m_train))
        error_val[i] = np.sum(np.square( h_val - yval ))/(2*m_val)
    return (lmda_vec, error_train, error_val)

lmda_vec, error_train, error_val = validation_curve( X_poly, y, X_poly_val, yval )
In [25]:
a = np.arange(1,error_train.size + 1)
plt.plot( lmda_vec, error_train, '-b', label = 'Train' )
plt.plot( lmda_vec, error_val, '-g', label = 'Cross Validation' )
plt.axis([0, np.amax(lmda_vec), 0, 20])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.legend()
plt.title('Validation curve')
plt.show()

lmda = 3
theta_trained = trainLinearReg( X_poly, y, lmda ).x.reshape(X_poly.shape[1], 1)
h_test = np.dot( X_poly_test, theta_trained )
test_error = np.sum(np.square( h_test - ytest ))/(2*Xtest.shape[0])
print(f'test error = {test_error}')