import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def run_frozen_lake(name_of_save_file, episodes, seed, learning_rate_a = 0.9, discount_factor_g = 0.9, epsilon=1, tau = 1.0, slippery=False, strategy="greedy", is_training=True, render=False):
   env = gym.make(
      'FrozenLake-v1',            
      map_name="8x8", 
      is_slippery=slippery, 
      render_mode='human' if render else None)
   env.reset(seed=seed)

   if(is_training):
      q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
   else:
      f = open(name_of_save_file + '.pkl', 'rb')
      q = pickle.load(f)
      f.close()

   epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
   rng = np.random.default_rng()   # random number generator

   rewards_per_episode = np.zeros(episodes)

   for i in range(episodes):
      state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
      terminated = False      # True when fall in hole or reached goal
      truncated = False       # True when actions > 200

      while(not terminated and not truncated):
         if strategy == "greedy":
            if is_training and rng.random() < epsilon:
               action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
               action = np.argmax(q[state,:])

         elif strategy == "boltzman":
            if is_training and rng.random() < epsilon:
               action = rng.choice(env.action_space.n, p=np.exp(q[state, :] / tau) / np.sum(np.exp(q[state, :] / tau)))
            else:
               action = np.argmax(q[state,:])

         new_state,reward,terminated,truncated,_ = env.step(action)

         if is_training:
               q[state,action] = q[state,action] + learning_rate_a * (
                  reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
               )

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

   if is_training:
      f = open(name_of_save_file + ".pkl","wb")
      pickle.dump(q, f)
      f.close()

# if __name__ == '__main__':
   # run("frozen_lake8x8", 10000, 42, is_training=True, render=True, strategy="greedy")

   #run(1000, is_training=True, render=False)