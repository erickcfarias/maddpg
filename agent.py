import numpy as np
import random
from collections import namedtuple, deque

from model import Critic, Actor
import torch
from torch import autograd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from random_process import *
from experience_replay import *


class AgentZero:
    def __init__(self, seed, env, brain_name, state_size, action_size, gamma):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = random.seed(seed)
        self.env = env
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        num_agents=2,
        gamma=0.99,
        seed=7,
        fc1=400,
        fc2=300,
        update_times=10,
        update_every=20,
        buffer_size=int(1e6),
        batch_size=512,
        critic_lr=1e-4,
        actor_lr=1e-3,
        weight_decay=1.e-5,
        tau=1e-3
        ):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.seed = random.seed(seed)
        self.n_seed = np.random.seed(seed)
        self.num_agents = num_agents
        self.update_times = update_times
        self.update_every = update_every
        self.batch_size = batch_size
        self.tau = tau
        self.eps = 1.

        # critic local and target network (Q-Learning)
        self.critic_local = Critic(
            state_size, action_size, fc1, fc2, seed).to(self.device)

        self.critic_target = Critic(
            state_size, action_size, fc1, fc2, seed).to(self.device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # actor local and target network for player 1
        self.actor_local_p1 = Actor(
            state_size, action_size, fc1, fc2, seed).to(self.device)
        self.actor_target_p1 = Actor(
            state_size, action_size, fc1, fc2, seed).to(self.device)
        self.actor_target_p1.load_state_dict(self.actor_local_p1.state_dict())

        # actor local and target network for player 2
        self.actor_local_p2 = Actor(
            state_size, action_size, fc1, fc2, seed).to(self.device)
        self.actor_target_p2 = Actor(
            state_size, action_size, fc1, fc2, seed).to(self.device)
        self.actor_target_p2.load_state_dict(self.actor_local_p2.state_dict())

        # optimizer for critic and actor network
        self.optimizer_critic = optim.Adam(
            self.critic_local.parameters(), lr=critic_lr,
            weight_decay=weight_decay)
        self.optimizer_actor_p1 = optim.Adam(
            self.actor_local_p1.parameters(), lr=actor_lr)
        self.optimizer_actor_p2 = optim.Adam(
            self.actor_local_p2.parameters(), lr=actor_lr)

        # Replay memory
        self.memory_p1 = ReplayBuffer(
            self.action_size, buffer_size, self.batch_size, seed)
        self.memory_p2 = ReplayBuffer(
            self.action_size, buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.a_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        for i in range(self.num_agents):

            all_state = np.concatenate((state[i], state[1-i]))
            all_actions = np.concatenate((action[i], action[1-i]))
            all_next_state = np.concatenate((next_state[i], next_state[1-i]))

            if i == 0:
                self.memory_p1.add(
                    state[i], all_state, action[i], all_actions,
                    reward[i], next_state[i], all_next_state, done[i])
            else:
                self.memory_p2.add(
                    state[i], all_state, action[i], all_actions,
                    reward[i], next_state[i], all_next_state, done[i])

        # Learn every update_every steps.
        self.t_step += 1
        if self.t_step % self.update_every == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory_p1) > self.batch_size:
                for i in range(self.update_times):
                    experiences_p1 = self.memory_p1.sample()
                    experiences_p2 = self.memory_p2.sample()
                    self.learn([experiences_p1, experiences_p2])

    def act(self, state, training=True):
        """Returns continous actions values for all action for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """

        state = torch.from_numpy(state).float().detach().to(device)
        #print(state.shape,"act")

        self.eps *= .99 # annealing the epsilon is good for convergence
        epsilon = self.eps

        self.actor_local_p1.eval()
        self.actor_local_p2.eval()
        with torch.no_grad():
            action1 = self.actor_local_p1(state[0].unsqueeze(0))
            action2 = self.actor_local_p2(state[1].unsqueeze(0))
            actions = torch.stack((action1.squeeze(0), action2.squeeze(0)))
        self.actor_local_p1.train()
        self.actor_local_p2.train()

        if training:
            #return np.clip(actions.cpu().data.numpy()+np.random.uniform(-1,1,(2,2))*epsilon,-1,1) #adding noise to action space
            r = np.random.random()
            if r <= epsilon:
                return np.random.uniform(-1, 1, (2, 2))
            else:
                # epsilon greedy policy
                return np.clip(actions.cpu().data.numpy(), -1, 1)
        else:
            return np.clip(actions.cpu().data.numpy(), -1, 1)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        for _, experience in enumerate(experiences):
            if _ == 0:
                states, all_state, action, all_actions, rewards, next_state, all_next_state, dones = experience

                all_next_actions = self.actor_target_p1(
                    all_next_state.view(self.batch_size*2, -1)).view(self.batch_size, -1)

                critic_target_input = torch.cat((all_next_state, all_next_actions.view(
                    self.batch_size*2, -1)[1::2]), dim=1).to(self.device)
                with torch.no_grad():
                    Q_target_next = self.critic_target(
                        critic_target_input, all_next_actions.view(self.batch_size*2, -1)[::2])
                Q_targets = rewards + (self.gamma * Q_target_next * (1-dones))

                critic_local_input = torch.cat(
                    (all_state, all_actions.view(self.batch_size*2, -1)[1::2]), dim=1).to(self.device)
                Q_expected = self.critic_local(critic_local_input, action)

                #critic loss
                l1_loss = torch.nn.SmoothL1Loss()

                loss = l1_loss(Q_expected, Q_targets.detach())

                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
                self.optimizer_critic.step()

                #actor loss

                action_pr_self = self.actor_local_p1(states)
                action_pr_other = self.actor_local_p1(
                    all_next_state.view(self.batch_size*2, -1)[1::2]).detach()

                #critic_local_input2=torch.cat((all_state,torch.cat((action_pr_self,action_pr_other),dim=1)),dim=1)
                critic_local_input2 = torch.cat((all_state, action_pr_other), dim=1)
                p_loss = -self.critic_local(critic_local_input2, action_pr_self).mean()

                self.optimizer_actor_p1.zero_grad()
                p_loss.backward()

                self.optimizer_actor_p1.step()

                # ------------------- update target network ------------------- #
                self.tau = min(5e-1, self.tau*1.001)  # ablation: + 1000 eps to converge with tau = 1e-3
                self.soft_update(self.critic_local, self.critic_target, self.tau)
                self.soft_update(self.actor_local_p1, self.actor_target_p1, self.tau)
            else:
                states, all_state, action, all_actions, rewards, next_state, all_next_state, dones = experience

                all_next_actions = self.actor_target_p2(
                    all_next_state.view(self.batch_size*2, -1)).view(self.batch_size, -1)

                critic_target_input = torch.cat((all_next_state, all_next_actions.view(
                    self.batch_size*2, -1)[1::2]), dim=1).to(self.device)
                with torch.no_grad():
                    Q_target_next = self.critic_target(
                        critic_target_input, all_next_actions.view(self.batch_size*2, -1)[::2])
                Q_targets = rewards + (self.gamma * Q_target_next * (1-dones))

                critic_local_input = torch.cat(
                    (all_state, all_actions.view(self.batch_size*2, -1)[1::2]), dim=1).to(self.device)
                Q_expected = self.critic_local(critic_local_input, action)

                #critic loss
                l1_loss = torch.nn.SmoothL1Loss()

                loss = l1_loss(Q_expected, Q_targets.detach())

                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_local.parameters(), 1)
                self.optimizer_critic.step()

                #actor loss

                action_pr_self = self.actor_local_p2(states)
                action_pr_other = self.actor_local_p2(
                    all_next_state.view(self.batch_size*2, -1)[1::2]).detach()

                #critic_local_input2=torch.cat((all_state,torch.cat((action_pr_self,action_pr_other),dim=1)),dim=1)
                critic_local_input2 = torch.cat(
                    (all_state, action_pr_other), dim=1)
                p_loss = - \
                    self.critic_local(critic_local_input2,
                                      action_pr_self).mean()

                self.optimizer_actor_p2.zero_grad()
                p_loss.backward()

                self.optimizer_actor_p2.step()

                # ------------------- update target network ------------------- #
                # ablation: + 1000 eps to converge with tau = 1e-3
                self.tau = min(5e-1, self.tau*1.001)
                self.soft_update(self.critic_local,
                                 self.critic_target, self.tau)
                self.soft_update(self.actor_local_p2,
                                 self.actor_target_p2, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
    
    def train(self, env, brain_name, n_episodes=5000):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
        """
        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        scores = []
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
                action = self.act(state)
                env_info = env.step(np.stack(action))[brain_name]
                next_state = env_info.vector_observations   # get the next state
                reward = env_info.rewards                   # get the reward

                done = env_info.local_done
                self.step(state, action, reward, next_state, done)
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
                torch.save(self.actor_local_p1.state_dict(),
                        'checkpoint_actor_p1.pth')
                torch.save(self.actor_local_p1.state_dict(),
                        'checkpoint_actor_p2.pth')
                break
        return scores
