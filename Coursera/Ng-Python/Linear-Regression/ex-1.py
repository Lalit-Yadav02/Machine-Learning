#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 00:36:44 2018

@author: lality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

"""
%matplotlib inline

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')
"""

# A 5x5 Identity Matrix

def warmUpExercise():
    return(np.identity(5))
    
warmUpExercise()

# Linear Regression with one variable
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]), data[:,0]]
y = np.c_[data[:,1]]

