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
        hs = propagateForward(row, thetas)[-1][1]
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

def propagate_forward(row, Thetas):
    features = row
    zs_as_per_layer = []
    for i in range(len(thetas)):
        theta = Thetas[i]
        z = theta.dot(features).reshpae((theta.shape[0], 1))
        a = expit(z)
        zs_as_per_layer.append((z, a))
        if i == len(Thetas) - 1:
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1)
        features = a


thetas = [Theta1, Theta2]
cost = compute_cost(flatten_params(thetas), flatten_X(X), y)
regularized_cost = compute_cost(flatten_params(thetas), flatten_X(X), y, 1.0)

# Implementing Backpropagation

# Sigmoid Gradient
def sigmoid_gradient(z):
    dummy = expit(z)
    return dummy * (1 - dummy)




#####
def genRandThetas():
    epsilon_init = 0.12
    theta1_shape = (hidden_layer_size, input_layer_size+1)
    theta2_shape = (output_layer_size, hidden_layer_size+1)
    rand_thetas = [ np.random.rand( *theta1_shape ) * 2 * epsilon_init - epsilon_init, \
                    np.random.rand( *theta2_shape ) * 2 * epsilon_init - epsilon_init]
    return rand_thetas

def backPropagate(mythetas_flattened,myX_flattened,myy,mylambda=0.):
    
    # First unroll the parameters
    mythetas = reshapeParams(mythetas_flattened)
    
    # Now unroll X
    myX = reshapeX(myX_flattened)

    #Note: the Delta matrices should include the bias unit
    #The Delta matrices have the same shape as the theta matrices
    Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
    Delta2 = np.zeros((output_layer_size,hidden_layer_size+1))

    # Loop over the training points (rows in myX, already contain bias unit)
    m = n_training_samples
    for irow in xrange(m):
        myrow = myX[irow]
        a1 = myrow.reshape((input_layer_size+1,1))
        # propagateForward returns (zs, activations) for each layer excluding the input layer
        temp = propagateForward(myrow,mythetas)
        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]
        tmpy = np.zeros((10,1))
        tmpy[myy[irow]-1] = 1
        delta3 = a3 - tmpy 
        delta2 = mythetas[1].T[1:,:].dot(delta3)*sigmoidGradient(z2) #remove 0th element
        a2 = np.insert(a2,0,1,axis=0)
        Delta1 += delta2.dot(a1.T) #(25,1)x(1,401) = (25,401) (correct)
        Delta2 += delta3.dot(a2.T) #(10,1)x(1,25) = (10,25) (should be 10,26)
        
    D1 = Delta1/float(m)
    D2 = Delta2/float(m)
    
    #Regularization:
    D1[:,1:] = D1[:,1:] + (float(mylambda)/m)*mythetas[0][:,1:]
    D2[:,1:] = D2[:,1:] + (float(mylambda)/m)*mythetas[1][:,1:]
    
    return flattenParams([D1, D2]).flatten()

#Actually compute D matrices for the Thetas provided
flattenedD1D2 = backPropagate(flattenParams(myThetas),flattenX(X),y,mylambda=0.)
D1, D2 = reshapeParams(flattenedD1D2)

def checkGradient(mythetas,myDs,myX,myy,mylambda=0.):
    myeps = 0.0001
    flattened = flattenParams(mythetas)
    flattenedDs = flattenParams(myDs)
    myX_flattened = flattenX(myX)
    n_elems = len(flattened) 
    #Pick ten random elements, compute numerical gradient, compare to respective D's
    for i in xrange(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems,1))
        epsvec[x] = myeps
        cost_high = computeCost(flattened + epsvec,myX_flattened,myy,mylambda)
        cost_low  = computeCost(flattened - epsvec,myX_flattened,myy,mylambda)
        mygrad = (cost_high - cost_low) / float(2*myeps)
        print "Element: %d. Numerical Gradient = %f. BackProp Gradient = %f."%(x,mygrad,flattenedDs[x])