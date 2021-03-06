#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:34:09 2021

@author: hshahab, ssahmad, rajasafi, mmmirza, anumm, jricoari
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read in distance matrix
D = pd.read_csv('five_d.csv', index_col=None, header=None).values
n = len(D)
cities = range(n)
print(D)

#Initialize Gurobi model
m = gp.Model()

#Add variables x
x = m.addVars(cities, cities, vtype = GRB.BINARY) #auxiliary variable if whether city i and j are connected along the tour

#Set objective function to minimize the total tour length
m.setObjective(sum(sum(D[i,j] * x[i,j] for i in cities) for j in cities))
# m.modelSense = GRB.MINIMIZE

#Add constraints

#Constraint #1
for i in cities:
    for j in cities:
        m.addConstr(x[i,j] == x[j,i])

#Constraint #2
for i in cities:
    for j in cities:
        m.addConstr(x[i,i] == 0)


#Constraint #3
for i in cities:
    m.addConstr(sum(x[i,j] for j in cities) == 2)

#Constraint #4 
# Helper function for the subtour elimination constraint
def get_power_set(s):
  power_set=[[]]
  for elem in s:
    # iterate over the sub sets so far
    for sub_set in power_set:
      # add a new subset consisting of the subset at hand added elem
      power_set=power_set+[list(sub_set)+[elem]]
  return power_set

power_set = get_power_set(cities)

# Sub-tour elimination constraint
for S in power_set:
    arcs = gp.LinExpr()
    if len(S) > 0 and len(S) < n:
        for i in S:
            for j in S:
                arcs += x[i,j]
        m.addConstr(arcs <= 2 * (len(S) - 1))
        

#Optimize Model
m.optimize()

print(m.ObjVal)

#sequence of cities 
for i in cities:
    for j in cities:
       if x[i,j].x == 1:
           print( 'from n = ', i, 'to n =', j, 'distance =', D[i,j])
        

