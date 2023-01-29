#Imports
import numpy as np
import pandas as pd
from UtilityMethods_in import utils
import matplotlib.pyplot as plt
import time
import os
import math
import pickle
import sys
import random

# temp = sys.argv[1:]
# RUN_NUMBER = int(temp[0])

RUN_NUMBER = 10 #Change this field to set the seed for the experiment.

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#RUN_NUMBER = 0

#Initialize:(Better with error handling like 'with' statement in 'open' operation)
f = open('model-in.pckl', 'rb')
[NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, DELTA] = pickle.load(f)
f.close()
print(NUMBER_EPISODES)
#read optimal policy from pre-trained model
f = open('solution-in.pckl', 'rb')
[opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon] = pickle.load(f)
f.close()


#read safe baseline policy from pre-trained model
f = open('base-in.pckl', 'rb')
[pi_b, val_b, cost_b, q_b] = pickle.load(f)
f.close()

EPS = 1

M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2

#np.random.seed(RUN_NUMBER)

#maximum permissible value for safe baseline policy
Cb = cost_b[0, 0]

print(CONSTRAINT - Cb)

#K0, number of episdoes that use safe baseline policy firstly for no solution concern
K0 = N_STATES**3*EPISODE_LENGTH**3/((CONSTRAINT - Cb)**2) 
print('K0: ', K0)

NUMBER_EPISODES = int(NUMBER_EPISODES) #ensure data type
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)

STATES = np.arange(N_STATES)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
ConRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))

NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))

# !! L not works here, the original below value should be used on formula about Rk and Ck
L = math.log(2 * N_STATES**2 * EPISODE_LENGTH * NUMBER_EPISODES / DELTA)#math.log(2 * N_STATES * EPISODE_LENGTH * NUMBER_EPISODES * N_STATES**2 / DELTA)

for sim in range(NUMBER_SIMULATIONS):
    util_methods = utils(EPS, DELTA, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,Cb)
    ep_count = np.zeros((N_STATES, N_STATES)) #I(s,a), I is the indicator function
    ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES)) #I(s,a,s')
    objs = []
    cons = []
    start_time = time.time()#record duration
    is_fixed = False
    cnt = 0
    #these variables are used for fixed method
    pi_f = np.zeros((N_STATES, EPISODE_LENGTH, N_STATES)) #[s,h,a], val_f, cost_f, log, q_f
    q_f = np.zeros((N_STATES,EPISODE_LENGTH, N_STATES)) #q_policy
    val_f = np.zeros((N_STATES, EPISODE_LENGTH)) #value
    cost_f = np.zeros((N_STATES,EPISODE_LENGTH)) #cost

    for episode in range(NUMBER_EPISODES):
        if episode <= K0:
            #use safe baseline policy for first K0 episodes
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b
            util_methods.setCounts(ep_count_p, ep_count) #number of times the pair (s,a) and (s,a,s') was observed
            util_methods.update_empirical_model(0) #update P_hat according to number of occurances
            util_methods.compute_confidence_intervals(L, 1) #compute confidence bound beta
        else:
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0)
            util_methods.compute_confidence_intervals(L, 0)
            if is_fixed == False: #update policy every 100 episodes
                #print('Update Policy on Epoch{}'.format(episode))
                pi_f, val_f, cost_f, log, q_f = util_methods.compute_extended_LP(0, Cb)
                is_fixed = True
            cnt += 1
            if cnt % 100 == 0:
                is_fixed = False
            pi_k = pi_f
            val_k = val_f
            cost_k = cost_f
            q_k = q_f
            if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that
                # pi_k = pi_b
                # val_k = val_b
                # cost_k = cost_b
                # q_k = q_b
                print(log)

        #calculate safe objective regret after K learning episodes
        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[0, 0] - opt_value_LP_con[0, 0]) #r = v_pi_k(P) - v_pi*(P)
            ConRegret2[sim, episode] = max(0, cost_k[0, 0] - CONSTRAINT)
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = 1 #no solution, not a safe policy
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[0, 0] - opt_value_LP_con[0, 0])
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + max(0, cost_k[0, 0] - CONSTRAINT)
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1
        
        #reset every episode
        ep_count = np.zeros((N_STATES, N_STATES))
        ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES))

        s = 0
        for h in range(EPISODE_LENGTH):
            prob = pi_k[s, h, :]
            #if sum(prob) != 1:
            #    print(s, h)
            #    print(prob)
            a = int(np.random.choice(STATES, 1, replace = True, p = prob)) #select action randomly according to p
            next_state, rew, cost = util_methods.step(s, a, h)
            ep_count[s, a] += 1 #update I(s,a)
            ep_count_p[s, a, next_state] += 1 #update I(s,a,s')
            s = next_state
        if episode != 0 and (episode+1)%5000 == 0: #decrease from 5e4 to 5e3 for better debug
            duration = time.time() - start_time
            print('Epoch {:d} Duration {:.3f} seconds.'.format(episode + 1, duration))
            filename = 'opsrl-in' + str(RUN_NUMBER) + '.pckl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []
        
print('Finished Training.')
ObjRegret_mean = np.mean(ObjRegret2, axis = 0)
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

#print(NUMBER_INFEASIBILITIES)

#print(util_methods.NUMBER_OF_OCCURANCES[0])


title = 'OPSRL' + str(RUN_NUMBER)
plt.figure()
plt.plot(range(NUMBER_EPISODES), ObjRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret')
plt.title(title)
plt.show()

time = np.arange(1, NUMBER_EPISODES+1)
squareroot = [int(b) / int(m) for b,m in zip(ObjRegret_mean, np.sqrt(time))]

plt.figure()
plt.plot(range(NUMBER_EPISODES),squareroot)
#plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret square root curve')
plt.title(title)
plt.show()

plt.figure()
plt.plot(range(NUMBER_EPISODES), ConRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ConRegret_mean - ConRegret_std, ConRegret_mean + ConRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Constraint Regret')
plt.title(title)
plt.show()

