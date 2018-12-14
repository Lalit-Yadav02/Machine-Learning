#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 07:19:34 2018

@author: lality

algorithm: Eclat Association Rule
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = [[str(dataset.values[i, j]) for j in range(dataset.shape[1]) if str(dataset.values[i, j]) != 'nan'] for i in range(dataset.shape[0])]

items = list()
for t in transactions:
    for x in t:
        if not x in items:
            items.append(x)
eclat = list()
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        eclat.append([(items[i], items[j]), 0])
        
for p in eclat:
    for t in transactions:
        if (p[0][0] in t) and (p[0][1] in t):
            p[1] += 1
    p[1] /= len(transactions)
    
eclat_df = pd.DataFrame(eclat, columns = ['rule', 'support']).sort_values(by = 'support', ascending = False)