#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 07:03:02 2018

@author: lality

algorithm: Apriori Association Rule
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header= None)
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1])])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

the_rules = []
for result in results:
    the_rules.append({
                'rule': ','.join(result.items),
                'support': result.support,
                'confidence': result.ordered_statistics[0].confidence,
                'lift': result.ordered_statistics[0].lift
            })
df = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])