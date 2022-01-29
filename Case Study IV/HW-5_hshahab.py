#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:15:57 2021

@author: hajrashahab
collaboration: : Li-Hsin Lin (lihsinl)
"""

#import packages 
from gurobipy import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Read in csv files as arrays 
f_composition= pd.read_csv('Pb1_composition.csv', index_col=None, header=None).values
composition = f_composition.astype(np.float)

energy = pd.read_csv('Pb1_energy.csv', index_col=None, header=None).values.reshape(-1)
energy = list(energy)

f_groups = pd.read_csv('Pb1_mapping.csv', index_col=None, header=None).values

price = np.genfromtxt('Pb1_price.csv', dtype=str, delimiter=',', encoding='utf-8-sig')
price = list(price)


#Setup indices 
f_items= range(len(composition)) #i = 1,...1007 (food items)
#print(f_items)
nutrients = range(10)  #k = 1,...10 (nutrients)  
#print(nutrients)
groups = range(6) #j = 1,...6(food groups)
#print(groups)


print('\nQ2') 

#Initialize Gurobi Model 
m = Model("Nutrition Policy")


#Setup Decision Variables 
#energy intake decision variables: e[i,j] captures the food item i for food group j  
x = m.addVars(f_items, lb = 0, ub=GRB.INFINITY)
q1 = [20000, 400, 7, 6.5, 0.57, 20, 0.7, 1.1, 0.050, 0.0005]
m1 = [0.011, 0.003, 0.35, 0.04, 0.009, 0.005]
m2 = [0.251, 0.087, 0.75, 0.33, 0.102, 0.085]


#set up two variables to capture the two objectives 
z1 = m.addVar(lb = 0.0) #for daily energy intake
z2 = m.addVar(lb = 0.0) #for total daily cost 


energy_intake = LinExpr()
for i in f_items:
    energy_intake += energy[i] * x[i]


daily_cost = LinExpr()
for i in f_items:
    daily_cost += price[i] * x[i]


m.addConstr(z1 == energy_intake)
m.addConstr(z2 == daily_cost)


#Add constraints 

#constraint 1
m.addConstr(sum(composition[i,0] * x[i] for i in f_items) >= q1[0])
m.addConstr(sum(composition[i,1] * x[i] for i in f_items) >= q1[1])
m.addConstr(sum(composition[i,2] * x[i] for i in f_items) >= q1[2])
m.addConstr(sum(composition[i,3] * x[i] for i in f_items) >= q1[3])
m.addConstr(sum(composition[i,4] * x[i] for i in f_items) >= q1[4])
m.addConstr(sum(composition[i,5] * x[i] for i in f_items) >= q1[5])
m.addConstr(sum(composition[i,6] * x[i] for i in f_items) >= q1[6])
m.addConstr(sum(composition[i,7] * x[i] for i in f_items) >= q1[7])
m.addConstr(sum(composition[i,8] * x[i] for i in f_items) >= q1[8])
m.addConstr(sum(composition[i,9] * x[i] for i in f_items) >= q1[9])

    
#constraint 2
for k in groups:
    m.addConstr(m1[k] * sum(energy[i]*x[i] for i in f_items) <= sum(f_groups[i][k]*energy[i]*x[i] for i in f_items))
    m.addConstr(sum(f_groups[i][k]*energy[i]*x[i] for i in f_items) <= m2[k] * sum(energy[i]*x[i] for i in f_items))
    
#Optimize Model 
alpha_values = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]
daily_intake= np.zeros([len(alpha_values), 2])
    
for iteration in range(len(alpha_values)):
    alpha = alpha_values[iteration]
    m.setObjective(alpha * z1 + (1 - alpha) * z2, GRB.MINIMIZE)
    m.optimize()
    
    print ("total daily energy: ", energy_intake.getValue(),'Kcal')
    print ("total food cost: $", daily_cost.getValue())
    
    daily_intake[iteration, 0] = z1.x
    daily_intake[iteration, 1] = z2.x


# Plot Pareto frontier
print(daily_intake)
plt.scatter(daily_intake[:,0],daily_intake[:,1])
plt.xlabel('Diet Cost')
plt.ylabel('Daily Energy Intake')
plt.show()


#Part3: goal programming approach
print('\nQ3') 
q = Model("Goal_Nutrition")

#Decision Variables
x1 = q.addVars(f_items, lb = 0, ub=GRB.INFINITY) 

#Objective is to minimize the daily energy intake by each child and the daily cost of diet

# Set up two variables to capture the two objectives 
z1 = q.addVar(lb = 0.0) # for daily energy intake
z2 = q.addVar(lb = 0.0) # for total daily cost of diet

#energy intake calculation
g_energy = LinExpr()
for i in f_items:
    g_energy += energy[i] * x1[i]

#Daily price calculation
g_price = LinExpr()
for i in f_items:
    g_price += price[i] * x1[i]

q.addConstr(z1 == g_energy)
q.addConstr(z2 == g_price)

#Nutrients constraints:
q.addConstr(sum(composition[i,0] * x1[i] for i in f_items) >= q1[0])
q.addConstr(sum(composition[i,1] * x1[i] for i in f_items) >= q1[1])
q.addConstr(sum(composition[i,2] * x1[i] for i in f_items) >= q1[2])
q.addConstr(sum(composition[i,3] * x1[i] for i in f_items) >= q1[3])
q.addConstr(sum(composition[i,4] * x1[i] for i in f_items) >= q1[4])
q.addConstr(sum(composition[i,5] * x1[i] for i in f_items) >= q1[5])
q.addConstr(sum(composition[i,6] * x1[i] for i in f_items) >= q1[6])
q.addConstr(sum(composition[i,7] * x1[i] for i in f_items) >= q1[7])
q.addConstr(sum(composition[i,8] * x1[i] for i in f_items) >= q1[8])
q.addConstr(sum(composition[i,9] * x1[i] for i in f_items) >= q1[9])

#constraint 2
for k in groups:
    q.addConstr(m1[k] * sum(energy[i]*x1[i] for i in f_items) <= sum(f_groups[i][k]*energy[i]*x1[i] for i in f_items))
    q.addConstr(sum(f_groups[i][k]*energy[i]*x1[i] for i in f_items) <= m2[k] * sum(energy[i]*x1[i] for i in f_items))
    
#Optimize model 
q.setObjective(z1, GRB.MINIMIZE)
z2.ub = 0.5
q.optimize()


#Output
print(q.objVal)


print ("total daily energy intake: ", g_energy.getValue(),'Kcal')
print ("total diet cost: $", g_price.getValue())    


  