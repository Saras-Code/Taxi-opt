#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import random


# In[7]:


def main():
    env = gym.make('Taxi-v3')
    
    #initializing quality table (q-table)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size,action_size))
    
    #hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0 #probability to explore
    decay_rate = 0.005
    
    #training varaiables
    num_episodes = 1000
    max_steps = 99

    # loop for training our AI
    for episode in range(num_episodes):
        
        #reset the environment 
        state = env.reset()
        done = False
        
        for step in range (max_steps):
            #exploring or exploiting
            if random.uniform(0,1) < epsilon:
                #exploration
                action = env.action_space.sample() #a random action
            else:
                #exploitation
                action = np.argmax(qtable[state,:])  #agent looks at Q-table and selects the action with the highest Q-value
                
            #choosing actions and observing rewards
            # every reset function (step) returns the 4 variables below
            new_state,reward,done,info = env.step(action)
            
            #Q-learning algorithm AKA Bellman Optimality Equation
            qtable[state,action]= qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            
            #Update to new state
            state = new_state
            
            #If we are done then finish the episode
            if done == True:
                break
        
        #Decrease the epsilon
        epsilon = np.exp(-decay_rate*episode)
        
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")
        # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()
        
            


# In[ ]:




