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

data = loadmat('data/ex5data1.mat')
X = data['X']
y = data['y']

def plot_data(X, y, xlabel, ylabel, xmin, xmax, ymin, ymax, axes = None):
    plt.plot(X, y, 'rx')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

plot_data(X, y, 'Change in water level(x)', 'Water flowing out of the dam(y)', -50, 40, 0, 40)

########################
def plot_dataset( X, y, xlabel, ylabel, xmin, xmax, ymin, ymax, axes = None ):
    plt.plot( X, y, 'rx' )
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
plot_dataset( data['X'], data['y'], 'Change in water level (x)', 'Water flowing out of the dam (y)', -50, 40, 0, 40 )