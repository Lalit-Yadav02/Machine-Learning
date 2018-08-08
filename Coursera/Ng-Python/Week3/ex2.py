#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:31:43 2018

@author: lality
"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

# function to load data
def loaddata(file, delimiter):
    data = np.loadtxt(file, delimiter=delimiter)
    print("Dimensions: ", data.shape)
    return data

# Logistic Regression
data = loaddata('data/ex2data1.txt', ',')
X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

def plotdata(data, label_x, label_y, label_pos, label_neg, axes = None):
    pos = data[:,2] == 1
    neg = data[:,2] == 0
    
    if axes is None:
        axes = plt.gca()
        
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon = True, fancybox = True)
    
plotdata(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not Admitted')

# Sigmoid Function
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# cost function
def costfunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1 * (1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    
    if np.isnan(J[0]):
        return np.inf
    
    return J[0]

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1/m) * X.T.dot(h-y)
    return grad.flatten()

initial_theta = np.zeros(X.shape[1])
cost = costfunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)

res = minimize(costfunction, initial_theta, args=(X, y), method=None, jac=gradient, options={'maxiter': 400})
res

# Predict
def predict(theta, X, threshold = 0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
    
# Student with Exam 1 score 45 and Exam 2 score 85
# Predict using the optimized Theta values from above (res.x)
sigmoid(np.array([1, 45, 85]).dot(res.x.T))

p = predict(res.x, X) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

# Decision Boundary
plotdata(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not Admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max()
x2_min, x2_max = X[:,2].min(), X[:,2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

# Regularized logistic regression
data2 = loaddata('data/ex2data2.txt', ',')
X = data[:,0:2]
y = np.c_[data2[:,2]]

plotdata(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 2')
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:,0:2])
XX.shape


def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return np.inf
    return J[0]

initial_theta = np.zeros(XX.shape[1])
costFunctionReg(initial_theta, 1, XX, y)