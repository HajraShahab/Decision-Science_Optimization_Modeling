#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:55:32 2021

@author: hajrashahab
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Question #4

#Indices for areas and shelters
resi_areas = range(len(areas)) #200 residential areas 
print(resi_areas)

p_sites = range(len(shelters)) #40 potential sites 
print(p_sites)

#Initializing gurobi model 
m = gp.Model('max_dist')


#Setting up Decision Variables 
x = m.addVars(p_sites, vtype = GRB.BINARY) 
y = m.addVars(resi_areas, p_sites, vtype = GRB.BINARY) 
r = areas[:, 2]
c = shelters[:, 2]
s = m.addVar(vtype = GRB.CONTINUOUS) 

#Setting up Objective Function 
m.setObjective(s)
m.modelSense = GRB.MINIMIZE

#Setting up Constraints 

m.addConstr(sum(x[j] for j in p_sites) <= 10)

for j in p_sites:
        m.addConstr(sum(r[i] * y[i,j] for i in resi_areas) <= c[j] * x[j])

for i in resi_areas:
        m.addConstr(sum (y[i,j] for j in p_sites) == 1)
        
for i in resi_areas:
    for j in p_sites:
        m.addConstr(sum(distances[i,j] * y[i,j] for j in p_sites) <= s)
        
# Solve
m.optimize()

# Print optimal cost
print(m.objVal)

for i in resi_areas:
    for j in p_sites:
        if y[i,j].x == 1:
            print(i, j, round(distances[i,j],2))



#For model 2
dist2 = []

for i in resi_areas:
    for j in p_sites:
        if y[i,j].x == 1:
            dist2.extend([distances[i,j]] * int(r[i])) 
               
plt.hist(dist2, bins=10)
plt.xlabel("Distance")
plt.ylabel("No of Residents")
plt.show()
    