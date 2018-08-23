#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:21:54 2018

@author: lality
"""

# Importing Standard Libraray
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import scipy.misc
import random
import scipy.optimize #fmin_cg to train neural network
import itertools
from scipy.special import expit #Vectorized sigmoid function

# load MATLAB files
data = loadmat('data/ex4data1.mat')
X = data['X']
y = data['y']
X = np.insert(X, 0, 1, axis = 1)
print("X shape: Rows:- {}, Columns:- {}".format(X.shape, X[0].shape))
print("Y shape: {}, unique records:- {}".format(y.shape, np.unique(y)))

def get_data_img(row, width = 20, height = 20):
    square = row[1:].reshape(width, height)
    return square.T

def display_data(indices = None):
    width, height = 20, 20
    rows, cols = 10, 10
    if not indices:
        indices = random.sample(range(X.shape[0]), rows*cols)
    pic = np.zeros((height*rows, width*cols))
    irow, icol = 0, 0
    for idx in indices:
        if icol == cols:
            irow += 1
            icol = 0
        i_img = get_data_img(X[idx])
        pic[irow * height: irow*height+i_img.shape[0], icol* width: icol*width+i_img.shape[1]] = i_img
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage(pic)
    plt.imshow(img, cmap = cm.Greys_r)

display_data()


data = loadmat('data/ex4weights.mat')
Theta1, Theta2 = data['Theta1'], data['Theta2']

input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10
n_training_samples = X.shape[0]

def flatten_params(theta_matrix):
    """
    This function will flatten a matrix into numpy array
    """
    flatten_list = [theta.flatten() for theta in theta_matrix]
    combined = list(itertools.chain.from_iterable(flatten_list))
    assert len(combined) == (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_layer_size
    return np.array(combined).reshape((len(combined), 1))

def reshape_params(flatten_array):
    theta1 = flatten_array[:(input_layer_size + 1) * hidden_layer_size].reshape((hidden_layer_size, input_layer_size + 1))
    theta2 = flatten_array[(input_layer_size + 1) * hidden_layer_size:].reshape((output_layer_size, hidden_layer_size + 1))
    return [theta1, theta2]

def flatten_X(_X):
    return np.array(_X.flatten()).reshape((n_training_samples * (input_layer_size + 1), 1))

def reshape_X(_X):
    return np.array(_X).reshape((n_training_samples, input_layer_size + 1))

def compute_cost(theta_flatten, X_flatten, _y, _lambda = 0.):
    thetas = reshape_params(theta_flatten)
    _X = reshape_X(X_flatten)
    total_cost = 0.
    m = n_training_samples
    for irow in range(m):
        row = _X[irow]
        hs = propagate_forward(row, thetas)[-1][1]
        tmp_y = np.zeros((10, 1))
        tmp_y[_y[irow] - 1] = 1
        cost = -tmp_y.T.dot(np.log(hs)) - (1-tmp_y.T).dot(np.log(1-hs))
        total_cost += cost
    total_cost = float(total_cost) / m
    total_reg = 0.
    for theta in thetas:
        total_reg += np.sum(theta * theta)
    total_reg *= float(_lambda)/(2*m)
    
    return total_cost + total_reg

def propagate_forward(row, thetas):
    features = row
    zs_as_per_layer = []
    for i in range(len(thetas)):
        theta = thetas[i]
        z = theta.dot(features).reshpae((theta.shape[0], 1))
        a = expit(z)
        zs_as_per_layer.append((z, a))
        if i == len(thetas) - 1:
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1)
        features = a
       

thetas = [Theta1, Theta2]
cost = compute_cost(flatten_params(thetas), flatten_X(X), y)