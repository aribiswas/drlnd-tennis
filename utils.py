# -*- coding: utf-8 -*-

import torch
import numpy as np

class OUNoise:
    
    def __init__(self, size, mean=0, mac=0.15, var=0.1, varmin=0.01, decay=1e-6, seed=0):
        """
        Initialize Ornstein-Uhlenbech action noise.
        Parameters
        ----------
        size : list or numpy array
            Dimensions of the noise [a,b] where a is the batch size and b is the number of actions
        mean : number, optional
            Mean of the OU noise. The default is 0.
        mac : number, optional
            Mena attraction constant. The default is 0.15.
        var : number, optional
            Variance. The default is 0.1.
        varmin : TYPE, optional
            Minimum variance. The default is 0.01.
        decay : number, optional
            Decay rate of variance. The default is 1e-6.
        seed : number, optional
            Seed. The default is 0.
        """
        np.random.seed(seed)
        self.mean = mean * np.ones(size)
        self.mac = mac
        self.var = var
        self.varmin = varmin
        self.decay = decay
        self.x = np.zeros(size) #0.25 * np.random.rand(20,4)
        self.xprev = self.x
        self.step_count = 0
        
    def step(self):
        """
        Step the OU noise model by computing the noise and decaying variance.
        Returns
        -------
        noise : numpy array
            OU action noise.
        """
        r = self.x.shape[0]
        c = self.x.shape[1]
        self.x = self.xprev + self.mac * (self.mean - self.xprev) + self.var * np.random.randn(r,c)
        self.xprev = self.x
        dvar = self.var * (1-self.decay)
        self.var = np.maximum(dvar, self.varmin)
        self.step_count += 1
        return self.x
    
    
    
class ExperienceBuffer:
    
    def __init__(self, state_dim, act_dim, num_agents, max_len=1e6):
        """
        Initialize a replay memory for storing:
            States
            Actions
            Rewards
            Next states
            Dones
            State values 
            Next state values
            Log probabilities 
            Advantage estimates
            Discounted rewards-to-go
        
        All items are stored as numpy arrays.
        Parameters
        ----------
        state_dim : number
            Dimension of states.
        act_dim : number
            Dimension of actions.
        max_len : number
            Capacity of memory.
        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        
        # elements in the buffer will be stacked on top of another
        # dimension of each buffer will be max_len x <buffer_size>
        self.states = np.empty((self.max_len,self.state_dim * num_agents))
        self.actions = np.empty((self.max_len,self.act_dim * num_agents))
        self.rewards = np.empty((self.max_len,num_agents))
        self.next_states = np.empty((self.max_len,self.state_dim * num_agents))
        self.dones = np.empty((self.max_len,num_agents))
        
        self.last_idx = -1
        
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experiences to replay memory.
        """
           
        self.last_idx += 1
        if self.last_idx >= self.max_len:
            self.last_idx = 0
        
        self.states[self.last_idx] = state
        self.actions[self.last_idx] = action
        self.rewards[self.last_idx] = reward
        self.next_states[self.last_idx] = next_state
        self.dones[self.last_idx] = done
        
        
    def sample(self, batch_size, device='cpu'):
        """
        Get randomly sampled experiences.
        """
        # random indices
        batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        
        # convert to tensors
        states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
        actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
        rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
        next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
        dones_batch = torch.from_numpy(self.dones[batch_idxs]).float().to(device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch


    def __len__(self):
        """Return the current size of internal memory."""
        
        return self.last_idx + 1


# ------ UTILITY FUNCTIONS -------

def get_action(actor, state, noise, train=False):
    
    with torch.no_grad():
        action = actor.mu(state).numpy()
            
        # If in train mode then add noise
        if train:
            action += noise
        
        # clip the action, just in case
        action = np.clip(action, -1, 1)
        
        return action
    
    
def soft_update(target_model, model, factor=0.01):
    """
    Soft update target networks.
    """
    with torch.no_grad():
        for target_params, params in zip(target_model.parameters(), model.parameters()):
            target_params.data.copy_(factor * params + (1-factor) * target_params.data)   
    



