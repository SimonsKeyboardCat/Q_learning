import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def run_frozen_lake_test(name_of_save_file, episodes, epsilon=1, tau = 1.0, slippery=True, strategy="greedy", render=False):
   env = gym.make(
      'FrozenLake-v1',            
      map_name="8x8", 
      is_slippery=slippery, 
      render_mode='human' if render else None)
   
   f = open(name_of_save_file + '.pkl', 'rb')
   q = pickle.load(f)
   f.close()

   epsilon_decay_rate = 0.0001
   rng = np.random.default_rng()

   rewards_per_episode = np.zeros(episodes)

   for i in range(episodes):
      state = env.reset()[0]
      terminated = False      
      truncated = False 

      while(not terminated and not truncated):
         if strategy == "greedy":
               action = np.argmax(q[state,:])

         elif strategy == "boltzman":
               action = np.argmax(q[state,:])

         new_state,reward,terminated,truncated,_ = env.step(action)
         state = new_state

      epsilon = max(epsilon - epsilon_decay_rate, 0)

      if(epsilon==0):
         learning_rate_a = 0.0001

      if reward == 1:
         rewards_per_episode[i] = 1

   env.close()

   sum_rewards = np.zeros(episodes)
   for t in range(episodes):
      sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
   plt.plot(sum_rewards)

   return sum_rewards[-10:]