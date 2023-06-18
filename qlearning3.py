import time
import gym
import random
import numpy as np
import gym
import random
################################ set environment and variables ##############################

env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False,render_mode='ansi')
action_size = env.action_space.n  # 4 action
state_size = env.observation_space.n #16 state
qtable = np.zeros((state_size, action_size))

print("Init Qtable:\n ")
print(qtable)
total_episodes = 300        # Total episodes
learning_rate = 0.8         # Learning rate
max_steps = 20              # Max steps per episode
gamma = 0.5                 # Discounting rate


######################################### Game is on ##########################
print("training...")
for episode in range(total_episodes):
    state = env.reset()[0] # Reset the environment and init state is given
    step = 0
    done = False
    print("EPISODE: ",episode)
    for step in range(max_steps):
        #random action
        action = env.action_space.sample()
        env.render()
        new_state, reward, done,truncated, info = env.step(action)
        #Somehow, the environment does not give negative rewards for game over, so hack it:
        if done and reward == 0:
            reward = -5
        if new_state == state:
            reward = -1
        print("NEW STATE:",new_state,"REWARD:",reward)
#        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        qtable[state, action]=reward + gamma * np.max(qtable[new_state, :])
        print("QTABLE AT",state,qtable[state])
        state = new_state
        if done: 
            print("GAME OVER.\n\n")
            break
    print("new QTABLE")
    print(qtable)

env.reset()
env.close()
print("result is going on")
################################### evaluation ###############################
env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False,render_mode='human')
state = env.reset()[0]
step = 0
done = False
print("****************************************************")
for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state,:])
        new_state, reward, done,truncated, info = env.step(action)
        if done:
            break
        state = new_state
time.sleep(2)
env.reset()
env.close()
        











