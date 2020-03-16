from unityagents import UnityEnvironment

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import os

from agent import Agent as MA

print('Loading Environment...')
<<<<<<< HEAD
env = UnityEnvironment(
    file_name="/home/erickfarias/Documentos/bitbucket/RL_nanodegree/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64",
    no_graphics=True)
=======
os.chdir('/home/erickfarias/Documentos/bitbucket/RL_nanodegree/deep-reinforcement-learning/p3_collab-compet/')
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=True)
>>>>>>> dual-agent

# # get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initialize the Agent
print("Raising Agent...")


def select_agent(agent):
    if agent == 'ddpg':
        pass
    elif agent == 'maddpg':
        return MA(24, 2, 2, fc1=400, fc2=300, seed=0, update_times=10)
    else:
        print("wrong selection. select from 1. ddpg, 2. maddpg")
agent = select_agent('maddpg')

scores = []


def solve_environment(n_episodes=6000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    global scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[
            brain_name]  # reset the environment
        
        state = env_info.vector_observations

        score = 0
        t = 0
        reward_this_episode_1 = 0
        reward_this_episode_2 = 0
        while True:
            t = t+1
            action = agent.act(state)
            env_info = env.step(np.stack(action))[brain_name]
            next_state = env_info.vector_observations   # get the next state
            reward = env_info.rewards                   # get the reward

            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            reward_this_episode_1 += reward[0]
            reward_this_episode_2 += reward[1]

            if np.any(done):
                break

        score = max(reward_this_episode_1, reward_this_episode_2)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local_p1.state_dict(),
                       'trained_weights/checkpoint_actor_p1.pth')
            torch.save(agent.actor_local_p1.state_dict(),
                       'trained_weights/checkpoint_actor_p2.pth')
            break
    return


solve_environment()
