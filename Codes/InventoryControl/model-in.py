#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:00:01 2021

@author: Anonymous
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UtilityMethods_in import utils
import sys
#import gym
import pickle
import time
import pulp as p
import math
from copy import copy

#fixed cost when the inventory is brought in
K = 4

#holding cost s for storing inventory
def h(x):
    return x

#revenue function
def f(x):
    if x > 0:
        return 8*x/100
    return 0

#purchase cost when inventory is brought in
def O(x):
    if x > 0:
        return K + 2*x
    return 0

N_STATES = 6 #number of states, namely maximum capacity for storage
actions = {} #actions the agent can take

delta = 0.01 #constant used in formula

R = {} #reward
C = {} #cost
P = {} #probability of the model

demand = [0.3,0.2,0.2,0.2,0.05,0.05] #distribution of stochastic exogenous demand, range from 0 to 5

suffix = '-------------' #suffix for display concern

#permissible actions for each state s, i.e. actions[0] = [0,1,2,3,4,5], an agent can buy 0-5 items since no storage used 
for s in range(N_STATES):
    actions[s] = []
    for a in range(N_STATES - s):
        actions[s].append(a)

#compute cost, p
for s in range(N_STATES):
    l = len(actions[s])
    R[s] = np.zeros(l)
    C[s] = np.zeros(l)
    P[s] = {}
    for a in actions[s]:
        C[s][a] = O(a) + h(s+a)
        P[s][a] = np.zeros(N_STATES)
        for d in range(N_STATES):
            s_ = s + a - d
            if s_ < 0:
                s_ = 0
            # will never happen, max(s_) = max(a + s) - min(d) = N_STATES - 1 - 0
            # elif s_ > N_STATES - 1: 
            #    s_ = N_STATES - 1
                
            P[s][a][s_] += demand[d]
        R[s][a] = 0
        # for s_ in range(N_STATES):
        #     R[s][a] += P[s][a][s_]*f(s_)
        
for s in range(N_STATES):
    for a in actions[s]:        
      for d in range(N_STATES):
            s_ = max(0,s+a-d) #ditto, no min needs -> s = max(0, s + a - d)
            if s + a - d >= 0:
                R[s][a] += P[s][a][s_]*f(d)
            else:
                R[s][a] += 0
            
def print_P_matrix(): #print matrix P with better format
    print(suffix + ' Print P Matrix ' + suffix)
    for s in range(N_STATES):
        for a in actions[s]:
            print('[{}, {}]: {}'.format(s, a, P[s][a]))    
print_P_matrix()

#find r_max, c_max for normalization concern
r_max = R[0][0]
c_max = C[0][0]
for s in range(N_STATES):
    for a in actions[s]:
        if C[s][a] > c_max:
            c_max = C[s][a]
        if R[s][a] > r_max:
            r_max = R[s][a]
print(suffix + ' Max R ' + suffix + '\n', r_max)

#normalize C and R
for s in range(N_STATES):
    for a in actions[s]:
        C[s][a] = C[s][a]/c_max
        R[s][a] = R[s][a]/r_max

EPISODE_LENGTH = 7

CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5  #Constraint for safety baseline policy, change this if you want different baseline policy.

NUMBER_EPISODES = 5e4 #cut run time in half
NUMBER_SIMULATIONS = 1

EPS = 0.01
M = 0

util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,C_b) #finding optimal policy
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0)
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0)
#save model
f = open('solution-in.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,C_b,C_b) #set CONSTRAINT = C_b for finding safe baseline policy 
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
#save model
f = open('base-in.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()

f = open('model-in.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, delta], f)
f.close()

print('*******')
print(opt_value_LP_uncon[0, 0])
print(opt_value_LP_con[0, 0])
print(value_b[0, 0],cost_b[0,0])
