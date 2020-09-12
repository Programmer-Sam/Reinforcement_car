import envModule as environment
import math
import pygame
import random
import numpy as np
import gym
from gym import error, spaces, utils

env=environment.CarEnv()
env.reset()

LEARNING_RATE=1e-1
DISCOUNT=0.99

intervalNO=[30]*len(env.observation_space.high)#number of bins/intervals you want
intervalSize=(env.observation_space.high-env.observation_space.low)/intervalNO#vector of input/number of bins

load="1000"
file=f"qtables/{load}-qtable.npy"#f-string just places variables within the string {load}
q_table= np.load(file)

ep_rewards=[]

def findState(state):#change states into whole numbers
    discrete_state=(state-env.observation_space.low)/intervalSize
    return tuple(discrete_state.astype(np.int))

def GiveMaxScore():
    discrete_state=findState(env.reset())
    rewardT=0
    done=False
    while not done:
        action=np.argmax(q_table[discrete_state])
        new_state,reward,done=env.step(action)
        new_discrete_state=findState(new_state)
        rewardT+=reward
        env.render()
        if not done:
            policyReward=np.max(q_table[new_discrete_state])
            current_q=q_table[discrete_state+(action,)]
            new_q=(1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*policyReward)
            q_table[discrete_state+(action,)]=new_q
        discrete_state=new_discrete_state
    pygame.quit()
    return (rewardT)

print(GiveMaxScore())
