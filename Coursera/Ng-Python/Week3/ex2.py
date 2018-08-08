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
    z = np.array(range(1, z))
    return (1 / (1 + np.exp(-z)))
