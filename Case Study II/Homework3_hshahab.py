#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:43:13 2021

@author: hajrashahab
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Question no.2
path = 'Pb2_areas.csv' 
data = np.genfromtxt(path, dtype=str, delimiter=',', encoding='utf-8-sig')
areas = data.astype(np.float) 
print(areas) 


path = 'Pb2_shelters.csv' 
data = np.genfromtxt(path, dtype=str, delimiter=',', encoding='utf-8-sig')
shelters = data.astype(np.float) 
print(shelters) 


#Indices for areas and shelters
resi_areas = range(len(areas)) #200 residential areas 
print(resi_areas)

p_sites = range(len(shelters)) #40 potential sites 
print(p_sites)

#Initializing gurobi model 
m = gp.Model('shelter_sites')


#Setting up Decision Variables 
x = m.addVars(p_sites, vtype = GRB.BINARY) 
y = m.addVars(resi_areas, p_sites, vtype = GRB.BINARY) 
r = areas[:, 2]
c = shelters[:, 2]
distances = np.empty((200, 40), float)
for i in resi_areas:
    for j in p_sites:
        distances[i,j] = abs(areas[i,0] - shelters[j,0]) + abs(areas[i,1] - shelters[j,1])


#Setting up Objective Function 
m.setObjective(sum(sum(r[i] * distances[i,j] * y[i,j] for i in resi_areas) for j in p_sites))
m.modelSense = GRB.MINIMIZE

#Setting up Constraints 

m.addConstr(sum(x[j] for j in p_sites) <= 10)

for j in p_sites:
        m.addConstr(sum(r[i] * y[i,j] for i in resi_areas) <= c[j] * x[j])

for i in resi_areas:
        m.addConstr(sum (y[i,j] for j in p_sites) == 1)
        
# Solve
m.optimize()

# Print optimal cost
print(m.objVal)

for j in p_sites:
    for i in resi_areas:
        if y[i,j].x == 1:
            print(i,j, round(distances[i,j],2))


#Question #5

#For model 1
dist1 = []

for i in resi_areas:
    for j in p_sites:
        if y[i,j].x == 1:
            dist1.extend([distances[i,j]] * int(r[i])) 
               
plt.hist(dist1, bins=10)
plt.xlabel("Distance")
plt.ylabel("No of Residents")
plt.show()
    
    








