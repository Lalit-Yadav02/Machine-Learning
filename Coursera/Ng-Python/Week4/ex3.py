#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:07:33 2018

@author: lality
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.misc
import matplotlib.cm as cm
import random
from scipy.special import expit
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression



# Loading the datafile
data = loadmat('data/ex3data1.mat')
X = data['X']
y = data['y']
X = np.insert(X, 0, 1, axis=1)
print("'y' shape: %s. Distinct y: %s"%(y.shape, np.unique(y)))
print("'X' shape: %s. X[0] shape: %s"%(X.shape, X[0].shape))

# Visualize the data
def get_image_data(row):
    """
    row: single np array with shape 1x400
    creates and returns the image
    """
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T

def display_data(indices = None):
    """
    Picks 100 random rows from X, creates a 20x20 image from each row
    Put all images together in 10x10 grid
    """
    width, height = 20, 20
    nrows, ncols = 10, 10
    if indices is None:
        indices = random.sample(range(X.shape[0]), nrows * ncols)
    
    pic = np.zeros((height*nrows, width*ncols))
    
    irow, icol = 0, 0
    for index in indices:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = get_image_data(X[index])
        pic[irow*height:irow*height + iimg.shape[0], icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(7, 7))
    img = scipy.misc.toimage(pic)
    plt.imshow(img, cmap = cm.Greys_r)
    
display_data()

# Logistic Regression
def sigmoid(z):
    return(1/ (1 + np.exp(-z)))
    
def lrCostFunction(theta, X, y, _lambda):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y) + (_lambda/(2*m)) * np.sum(np.square(theta[1:])))
    if np.isnan(J[0]):
        return np.inf
    return J[0]

def lrGradientFunction(theta, X, y, _lambda):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1/m) * X.T.dot(h - y) + (_lambda/m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()

# One-vs-all-Classification
def oneVsAll(features, classes, n_labels, _lambda):
    initial_theta = np.zeros((X.shape[1],1))
    all_theta = np.zeros((n_labels, X.shape[1]))
    
    for c in np.arange(1, n_labels+1):
        res = minimize(lrCostFunction, initial_theta, args=(features, (classes == c)*1, _lambda), method=None, jac=lrGradientFunction, options={'maxiter': 50})
        all_theta[c - 1] = res.x
    return all_theta

theta = oneVsAll(X, y, 10, 0.1)

def predictOneVsAll(all_theta, features):
    probs = sigmoid(X.dot(all_theta.T))
    return np.argmax(probs, axis=1) + 1

pred = predictOneVsAll(theta, X)
print('Accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

# Multiclass Logistic Regression
clf = LogisticRegression(C=10, penalty='l2', solver='liblinear')
clf.fit(X[:,1:], y.ravel())

pred2 = clf.predict(X[:,1:])
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))

def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c[np.ones((data['X'].shape[0],1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
    
    return (np.argmax(a3, axis=1) + 1)

pred = predict(theta1, theta2, X)

###### Below ############
clf = LogisticRegression(C=10, penalty='l2', solver='liblinear')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
clf.fit(X[:,1:],y.ravel())

pred2 = clf.predict(X[:,1:])
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))

def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0],1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1)
    
pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))