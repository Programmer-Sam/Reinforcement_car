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
EPISODES=1000
SHOW_EVERY=200
SAVE_EVERY=1000
view_every=1
epsilon=0.5
startExploration=1
endExploration= EPISODES//1.2

intervalNO=[30]*len(env.observation_space.high)
intervalSize=(env.observation_space.high-env.observation_space.low)/intervalNO

q_table=np.random.uniform(low=-1,high=-1,size=(intervalNO+[env.action_space.n]))

ep_rewards=[]
aggr_ep_rewards={'episode':[],'avg':[],'min':[],'max':[]}

def findState(state):
    discrete_state=(state-env.observation_space.low)/intervalSize
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward=0
    discrete_state=findState(env.reset())
    done=False
    while not done:
        x=np.random.random()
        if x>epsilon:
            action=np.argmax(q_table[discrete_state])
        else:
            action=np.random.randint(0,env.action_space.n)
        new_state,reward,done=env.step(action)
        new_discrete_state=findState(new_state)
        episode_reward+=reward
        if episode//view_every==0:
            env.render()
        if not done:
            policyReward_q=np.max(q_table[new_discrete_state])
            current_q=q_table[discrete_state+(action,)]
            new_q=(1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*policyReward_q)
            q_table[discrete_state+(action,)]=new_q
        discrete_state=new_discrete_state
    if endExploration>=episode>=startExploration:
        epsilon-=(epsilon/(endExploration-startExploration))

    ep_rewards.append(episode_reward)
    if not episode%SHOW_EVERY:
        average_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['episode'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode:{episode} avg:{average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])} E:{epsilon}")
    if episode%SAVE_EVERY==0:
        np.save(f"qtables/{episode}-qtable2.npy", q_table)
env.close()
#pygame.quit()
