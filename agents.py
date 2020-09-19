# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DeterministicActor, QCritic
from utils import OUNoise, soft_update


# ========== DDPG Agent =============


class DDPGAgent:
    """
    Class for implementing a Deep Deterministic Policy Gradient (DDPG) agent.

    Attributes
    ----------
    actor : model.DeterministicActor
        Deterministic actor object..
    critic : model.QCritic
        Critic object.
    target_actor : model.DeterministicActor
        Target actor object.
    target_critic : model.QCritic
        Target critic object.
    buffer : utils.ExperienceBuffer
        Experience buffer object.

    Methods
    -------
    get_action(state, train=False):
        Get the action by sampling from the policy.
    step(state, action, reward, next_state, done, train=True):
        Step the agent, store experiences and learn.
    learn(experiences):
        Train the actor and critic networks from experiences.
        
    """
    
    def __init__(self, actor, critic, target_actor, target_critic, buffer, params):
        """
        Initialize a Deep Deterministic Policy Gradient (DDPG) agent.

        Parameters
        ----------
        actor : model.DeterministicActor
            Deterministic actor object..
        critic : model.QCritic
            Critic object.
        target_actor : model.DeterministicActor
            Target actor object.
        target_critic : model.QCritic
            Target critic object.
        buffer : utils.ExperienceBuffer
            Experience buffer object.
        params : dict
            Dictionary of DDPG hyperparameters containing the following keys:
                GAMMA        : Discount factor
                BATCH_SIZE   : Batch size for training.
                TAU          : Target network smooth factor.
                ACTOR_LR     : Learn rate for actor network.
                CRITIC_LR    : Learn rate for critic network.
                UPDATE_FREQ  : Frequency of target network updates.
                TRAIN_ITERS  : Number of training iterations per learn step.
                NOISE_MEAN   : Mean of the OU Noise.
                NOISE_MAC    : Mean attraction constant of OU Noise.
                NOISE_VAR    : Variance of OU Noise.
                NOISE_VARMIN : Minimum variance of OU Noise.
                NOISE_DECAY  : Decay rate of OU Noise variance.

        Returns
        -------
        None.

        """
        
        # parameters
        self.gamma = params["GAMMA"]
        self.batch_size = params["BATCH_SIZE"]
        self.tau = params["TAU"]
        self.actor_LR = params["ACTOR_LR"]
        self.critic_LR = params["CRITIC_LR"]
        self.update_freq = params["UPDATE_FREQ"]
        self.train_iters = params["TRAIN_ITERS"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.step_count = 0
        
        # initialize actor
        self.actor = actor
        self.target_actor = target_actor
        
        # initialize critic
        self.critic = critic
        self.target_critic = target_critic
        
        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_LR)
        
        # Experience replay
        self.buffer = buffer
        
        # Noise model
        self.noise_model = OUNoise(size=params["NOISE_SIZE"], mean=params["NOISE_MEAN"], 
                                   mac=params["NOISE_MAC"], var=params["NOISE_VAR"], 
                                   varmin=params["NOISE_VARMIN"], decay=params["NOISE_DECAY"])
        
        # initialize logs
        self.actor_loss_log = [0]
        self.critic_loss_log = [0]
        self.noise_log = [0]
        
    
    def get_action(self, state, train=False):
        """
        Get the action by sampling from the policy. If train is set to True 
        then the action contains added noise.

        Parameters
        ----------
        state : numpy.ndarray
            State of the environment.
        train : boolean, optional
            Flag for train mode. The default is False.

        Returns
        -------
        action : numpy.ndarray
            Action sampled from the agent's actor, with optional added noise. 
            The action is clipped within -1 and 1. 

        """
        
        with torch.no_grad():
            action = self.actor.action(state).cpu().numpy()
            
        # If in train mode then add noise
        if train:
            noise = self.noise_model.step()
            action += noise
            self.noise_log.append(self.noise_model.var)
        
        # clip the action, just in case
        action = np.clip(action, -1, 1)
        
        return action
        
    
    def step(self, state, action, reward, next_state, done, train=True):
        """
        Step the agent, store experiences and learn.

        Parameters
        ----------
        state : numpy.ndarray
            State of the environment.
        action : numpy.ndarray
            Action from the agent.
        reward : numpy.ndarray
            Reward for taking the action.
        next_state : numpy.ndarray
            Next states of the environment.
        done : numpy.ndarray
            Flag for termination.
        train : boolean, optional
            Flag for train mode. The default is True.

        Returns
        -------
        None.

        """
        
        # add experience to replay
        self.buffer.add(state, action, reward, next_state, done)
        
        # increase step count
        self.step_count += 1
        
        # learn from experiences
        if train and self.buffer.__len__() > self.batch_size:
            
            for _ in range(self.train_iters):
            
                # create mini batch for learning
                experiences = self.buffer.sample(self.batch_size)
                
                # train the agent
                self.learn(experiences)
            
    
    
    def learn(self, experiences):
        """
        Train the actor and critic networks from experiences.

        Parameters
        ----------
        experiences : List
            Sampled experiences from buffer.

        Returns
        -------
        None.

        """
        
        # unpack experience
        states, actions, rewards, next_states, dones = experiences
        
        # compute td targets
        with torch.no_grad():
            target_action = self.target_actor.action(next_states)
            targetQ = self.target_critic.Q(next_states,target_action)
            y = rewards + self.gamma * targetQ * (1-dones)
        
        # compute local Q values
        Q = self.critic.Q(states, actions)
        
        # critic loss
        critic_loss = F.mse_loss(Q,y)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # gradient clipping
        self.critic_optimizer.step()
        
        # freeze critic before policy loss computation
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # actor loss
        actor_loss = -self.critic.Q(states, self.actor.action(states)).mean()
        
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # gradient clipping
        self.actor_optimizer.step()
        
        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True
            
        # log the loss and noise
        self.actor_loss_log.append(actor_loss.detach().cpu().numpy())
        self.critic_loss_log.append(critic_loss.detach().cpu().numpy())
        
        # soft update target actor and critic
        if self.step_count % self.update_freq == 0:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
            





