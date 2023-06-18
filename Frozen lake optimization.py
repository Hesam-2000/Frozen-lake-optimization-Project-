import gym
import numpy as np
import random

env = gym.make('FrozenLake-v1')

def value_iteration(env):
    
        #set the number of iterations
        num_iterations = 1000
        
        #set the threshold number for checking the convergence of the value function
        threshold = 1e-20
        
        #set the discount factor
        gamma = 1.0
        
        #REMINDER. The discount factor essentially determines how much the reinforcement learning
        #cares about rewards in the distant future relative to those in the near future.
        #If γ=0, the agent will be completely myopic and only learn about actions that produce
        #If γ=1, the agent will evaluate each of its actions based on the sum total of all of i
        #now, we will initialize the value table, with the value of all states to zero
        value_table = np.zeros(env.observation_space.n)
        
        #for every iteration
        for i in range(num_iterations):
            
            #update the value table: every iteration, we use the updated value
            #table (state values) from the previous iteration
            updated_value_table = np.copy(value_table)
            
            #next, we compute the value function (state value) by taking the maximum of Q value
            
            #thus, for each state, we compute the Q values of all the actions in the state and
            #we update the value of the state as the one which has maximum Q value as shown bel
            for s in range(env.observation_space.n):
            
                Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
                                 for prob, s_, r, _ in env.P[s][a]])
                                        for a in range(env.action_space.n)]
                
                value_table[s] = max(Q_values)
        
        #after computing the value table, that is, value of all the states, we check whethe
        #difference between value table obtained in the current iteration and previous iter
        #less than or equal to a threshold value. If that conditon is true, then we break t
        #value table as our optimal value function as shown below:
        
            if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
                break
   
        return value_table

def extract_policy(value_table):
    
    #set the discount factor.
    
    gamma = 1
    
    #first, we initialize the policy with zeros, that is, first, we set the actions for all
    #be zero
    policy = np.zeros(env.observation_space.n)
    
    #Next, we compute the Q function using the optimal value function obtained from the
    #previous step. After computing the Q function, we can extract policy by selecting acti
    #maximum Q value. Since we are computing the Q function using the optimal value
    #function, the policy extracted from the Q function will be the optimal policy.
    
    #As shown below, for each state, we compute the Q values for all the actions in the sta
    #then we extract policy by selecting the action which has maximum Q value.
    
    #for each state
    for s in range(env.observation_space.n):
        
        #compute the Q value of all the actions in the state (again, we apply one of the Be
        Q_values = [sum([prob*(r + gamma * value_table[s_])
                         for prob, s_, r, _ in env.P[s][a]])
                            for a in range(env.action_space.n)]
        #extract policy by selecting the action which has maximum Q value
        policy[s] = np.argmax(np.array(Q_values))

optimal_value_function = value_iteration(env=env)
optimal_policy = extract_policy(optimal_value_function)

print(optimal_policy)

