#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 17:41:05 2021

@author: hajrashahab
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import files 
path = 'PB1_D_stochastic.csv' 
data = np.genfromtxt(path, dtype=str, delimiter=',', encoding='utf-8-sig')
demand = data.astype(np.float)

path = 'PB1_prob.csv' 
data = np.genfromtxt(path, dtype=str, delimiter=',', encoding='utf-8-sig')
prob = data.astype(np.float)

#setting up indices and parameters
s_type = range(len(demand)) #i = 1,…, N (N products = Economy, Economy+, Business & First Class)
#print(s_type)
scenarios = range(len(prob)) #j = 1,…, M (demand levels)
#print(scenarios)


#initialize gruobi model 
m = gp.Model("Airline Seats")

#setting variables 
avg_demand = []
for i in s_type:
    avg_demand.append(np.mean(demand[i, :]))
#print(avg_demand)

price = [400, 500, 800, 1000]
e = m.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0) #economy 
eplus = m.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0) #economy+ 
b = m.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0) #business class 
fc = m.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0) #first class 


#set objective function 
m.setObjective(sum(prob[j] * (400 * e[j] + 500 * eplus[j] + 800 * b[j] + 1000 * fc[j]) for j in scenarios))
m.modelSense = GRB.MAXIMIZE


#add constraints 
for j in scenarios:
    m.addConstr(e[j] + eplus[j] * 1.2 + b[j] * 1.5 + fc[j] * 2 <= 190)
    
#demand constraints 
for j in scenarios:
    m.addConstr(e[j] <= demand[0,j]) # economy 
    m.addConstr(eplus[j] <= demand[1,j]) # economy plus 
    m.addConstr(b[j] <= demand[2,j]) # business 
    m.addConstr(fc[j] <= demand[3,j]) # first class 

#optimize 
m.optimize()


for i in scenarios:
    print(e[j].x, eplus[j].x, b[j].x, fc[j].x)
    
#unmet demand 

unmet_demand = []
e_demand = []
eplus_demand = []
b_demand = []
fc_demand = []
cap_met = []
for j in scenarios:
    e_demand.append(e[j].x == demand[0,j])
    eplus_demand.append(eplus[j].x == demand[1,j])
    b_demand.append(b[j].x == demand[2,j])
    fc_demand.append(fc[j].x == demand[3,j])
    
    cap_met.append(round(e[j].x + eplus[j].x * 1.2 + b[j].x * 1.5 + fc[j].x * 2, 0))

unmet_demand.append(1000 - sum(e_demand))
unmet_demand.append(1000 - sum(eplus_demand))
unmet_demand.append(1000 - sum(b_demand))
unmet_demand.append(1000 - sum(fc_demand))
arr = np.array(cap_met)
unique, counts = np.unique (arr, return_counts=True)
print (unique, counts)

print(unmet_demand)
arr1 = np.array(unmet_demand)
unique, counts = np.unique (arr1, return_counts=True)
print (unique, counts)

#Part 2
m1 = gp.Model("Airline Seats")

### Setting up Decision Variable ###
e2 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
eplus2 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
b2 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
fc2 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)

e3 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
eplus3 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
b3 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
fc3 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)

e4 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
eplus4 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
b4 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)
fc4 = m1.addVars(scenarios, vtype=GRB.INTEGER, lb = 0.0)

#objective function
m1.setObjective(0.40 * (sum(prob[j] * (400 * e2[j] + 500 * eplus2[j] + 800 * b2[j] + 1000 * fc2[j]) for j in scenarios)) +
                0.30 * (sum(prob[j] * (400 * e3[j] + 500 * eplus3[j] + 600 * b3[j] + 700 * fc3[j]) for j in scenarios)) +
                0.30 * (sum(prob[j] * (400 * e4[j] + 420 * eplus4[j] + 600 * b4[j] + 700 * fc4[j]) for j in scenarios)))

m1.modelSense = GRB.MAXIMIZE

# supply/capacity constraint
for j in scenarios:
    m1.addConstr(e2[j] + eplus2[j] + b2[j] + fc2[j] <= 190)
    m1.addConstr(e3[j] + eplus3[j] + b3[j] + fc3[j] <= 190)
    m1.addConstr(e4[j] + eplus4[j] + b4[j] + fc4[j] <= 190)

    
#demand constraint
for j in scenarios:
    m1.addConstr(e2[j] <= demand[0,j]) 
    m1.addConstr(eplus2[j] <= demand[1,j]) 
    m1.addConstr(b2[j] <= demand[2,j]) 
    m1.addConstr(fc2[j] <= demand[3,j]) 
    
    m1.addConstr(e3[j] <= demand[0,j]) 
    m1.addConstr(eplus3[j] <= demand[1,j]) 
    m1.addConstr(b3[j] <= demand[2,j]) 
    m1.addConstr(fc3[j] <= demand[3,j]) 
    
    m1.addConstr(e4[j] <= demand[0,j]) 
    m1.addConstr(eplus4[j] <= demand[1,j]) 
    m1.addConstr(b4[j] <= demand[2,j]) 
    m1.addConstr(fc4[j] <= demand[3,j]) 
    
#optimize 
m.optimize()

#this code seems fine but it is constantly throwing an error hence leaving it in comments 
# for j in scenarios:
#     if e2[j].x < e[j].x:
#         print(e2[j], e[j])
#     if eplus2[j].x < eplus[j].x:
#         print(eplus2[j].x, eplus[j].x)
#     if b2[j].x < b[j].x:
#         print(b2[j].x, b[j].x)
#     if fc2[j].x < fc[j].x:
#         print(fc2[j].x, fc[j].x)


# for j in scenarios:
#     print(eplus2[j].x, eplus3[j].x, eplus4[j].x)
    




