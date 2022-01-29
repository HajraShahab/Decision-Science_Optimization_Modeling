#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 14:11:20 2021

@author: hajrashahab
"""

from gurobipy import *
import numpy as np
import csv
import os


#Preparing files for LP model
u = 'Pb1_requirements.csv'
data = np.genfromtxt(u, dtype=str, delimiter=',', encoding='utf-8-sig')
requirements = data.astype(np.float)
print(requirements)

v = 'Pb1_availability.csv'
data = np.genfromtxt(v, dtype=str, delimiter=',', encoding='utf-8-sig')
availability = data.astype(np.float)
print(availability)

x = 'Pb1_demand.csv'
data = np.genfromtxt(x, dtype=str, delimiter=',', encoding='utf-8-sig')
demand = data.astype(np.float)
print(demand)

y = 'Pb1_unitprofit.csv'
data = np.genfromtxt(y, dtype=str, delimiter=',', encoding='utf-8-sig')
unitprofit = data.astype(np.float)
print(unitprofit)

z = 'Pb1_holdingcost.csv'
data = np.genfromtxt(z, dtype=str, delimiter=',', encoding='utf-8-sig')
holdingcost = data.astype(np.float)
print(holdingcost)

# Indices
products = range(len(requirements))
print(products)



# Setting up model object
m = Model()


#Define variables, variable bounds, and objective separately
produce = m.addVars(months, products, name="Make") # quantity manufactured
hold = m.addVars(months, products, ub=max_inventory, name="Store") # quantity stored
sell = m.addVars(months, products, ub=max_sales, name="Sell") # quantity sold


m.setObjective(sum(sum(transportCosts[i,j] * x[i,j] for i in plants) for j in markets))
m.modelSense = GRB.MINIMIZE


for i in plants:
    for j in markets:
        m.addConstr(x[i,j] >= 0.0)


### Constraints ###

# Demand constraints
for j in markets:
  m.addConstr(sum(x[i,j] for i in plants) >= demand[j])

# Production constraints
# Note that the right-hand limit sets the production to zero if the plant
# is closed
for i in plants:
  m.addConstr(sum(x[i,j] for j in markets) <= capacity[i])
  
  
# Solve
m.optimize()

# Print optimal cost
print(m.objVal)

# Print optimal solution
    
for i in plants:
    for j in markets:
        print(i, j, x[i,j].x)
        
