# -*- coding: utf-8 -*-

import torch
import numpy as np

class OUNoise:
    """
    Class for implementing Ornstein-Uhlenbeck noise model. 

    Attributes
    ----------
    mean : number
        Noise mean.
    mac : number
        Mean attraction constant.
    var : number
        Noise variance.
    varmin : number
        Minimum variance.
    decay : number
        Variance decay rate.
    size : number
        Size of noise.

    Methods
    -------
    step():
        Step the OU noise model by computing the noise and decaying variance.
        
    """
    
    def __init__(self, size, mean=0, mac=0.15, var=0.1, varmin=0.01, decay=1e-6):
        """
        Initialize Ornstein-Uhlenbech action noise.

        Parameters
        ----------
        size : TYPE
            Size of noise.
        mean : number, optional
            Noise mean. The default is 0.
        mac : number, optional
            Mean attraction constant. The default is 0.15.
        var : number, optional
            Noise variance. The default is 0.1.
        varmin : number, optional
            Minimum variance. The default is 0.01.
        decay : number, optional
            Variance decay rate. The default is 1e-6.

        Returns
        -------
        None.

        """
        
        self.size   = size
        self.mean   = mean * np.ones(size)
        self.mac    = mac
        self.var    = var
        self.varmin = varmin
        self.decay  = decay
        self.x      = np.zeros(size)
        self.xprev  = self.x
        
        self.step_count = 0
        
        
    def step(self):
        """
        Step the OU noise model by computing the noise and decaying variance.

        Returns
        -------
        numpy.ndarray
            New noise.

        """
        
        self.x      = self.xprev + self.mac * (self.mean - self.xprev) + self.var * np.random.randn(self.size)
        self.xprev  = self.x
        self.var    = np.maximum(self.var * (1-self.decay), self.varmin)
        
        self.step_count += 1
        
        return self.x
    
    

class ExperienceBuffer:
    """
    Class for implementing a replay memory. 

    Attributes
    ----------
    mean : number
        Noise mean.
    mac : number
        Mean attraction constant.
    var : number
        Noise variance.
    varmin : number
        Minimum variance.
    decay : number
        Variance decay rate.
    size : number
        Size of noise.

    Methods
    -------
    step():
        Step the OU noise model by computing the noise and decaying variance.
        
    """
    
    def __init__(self, state_dim, act_dim, max_len=1e6):
        """
        Initialize a replay memory for storing experiences.

        Parameters
        ----------
        state_dim : number
            Size of states.
        act_dim : number
            Size of actions.
        max_len : number, optional
            Maximum length of buffer. The default is 1e6.

        Returns
        -------
        None.

        """
        
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.max_len    = max_len
        
        # elements in the buffer will be stacked on top of another
        self.states         = np.empty((self.max_len,self.state_dim))
        self.actions        = np.empty((self.max_len,self.act_dim))
        self.rewards        = np.empty((self.max_len,1))
        self.next_states    = np.empty((self.max_len,self.state_dim))
        self.dones          = np.empty((self.max_len,1))
        
        self.last_idx = -1
        
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experiences to replay memory.

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

        Returns
        -------
        None.

        """
           
        self.last_idx += 1
        if self.last_idx >= self.max_len:
            self.last_idx = 0
        
        self.states[self.last_idx]      = state
        self.actions[self.last_idx]     = action
        self.rewards[self.last_idx]     = reward
        self.next_states[self.last_idx] = next_state
        self.dones[self.last_idx]       = done
        
        
    def sample(self, batch_size, device='cpu'):
        """
        Get randomly sampled experiences.

        Parameters
        ----------
        batch_size : number
            Size of training batch.
        device : char, optional
            'cpu' or 'cuda'. The default is 'cpu'.

        Returns
        -------
        states_batch : torch.Tensor, torch.cuda.FloatTensor
            Mini batch of states.
        actions_batch : torch.Tensor, torch.cuda.FloatTensor
            Mini batch of actions.
        rewards_batch : torch.Tensor, torch.cuda.FloatTensor
            Mini batch of rewards.
        next_states_batch : torch.Tensor, torch.cuda.FloatTensor
            Mini batch of next_states.
        dones_batch : torch.Tensor, torch.cuda.FloatTensor
            Mini batch of dones.

        """
        # random indices
        batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        
        # convert to tensors
        states_batch        = convert_to_tensor(self.states[batch_idxs], device)
        actions_batch       = convert_to_tensor(self.actions[batch_idxs], device)
        rewards_batch       = convert_to_tensor(self.rewards[batch_idxs], device)
        next_states_batch   = convert_to_tensor(self.next_states[batch_idxs], device)
        dones_batch         = convert_to_tensor(self.dones[batch_idxs], device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch


    def __len__(self):
        """
        Return the current size of internal memory.

        Returns
        -------
        number
            Size of internal memory.

        """
        return self.last_idx + 1


# ------ UTILITY FUNCTIONS -------

def action_with_noise(actor, state, noise, bounds=[-1,1], train=False):
        """
        Get the action by sampling from the actor. If train is set to True 
        then the action contains added noise.

        Parameters
        ----------
        actor : model.DeterministicActor
            Deterministic actor object.
        state : numpy.ndarray
            State of the environment.
        bounds : list
            Upper and lower bound of action.
        train : boolean, optional
            Flag for train mode. The default is False.

        Returns
        -------
        action : numpy.ndarray
            Action sampled from the agent's actor, with optional added noise. 
            The action is clipped within -1 and 1. 

        """
        
        with torch.no_grad():
            action = actor.action(state).cpu().numpy()
            
        # If in train mode then add noise
        if train:
            action += noise
        
        # clip the action, just in case
        action = np.clip(action, bounds[0], bounds[1])
        
        return action

def sim_act(actor, state):
    """
    Get the action from the policy (actor), given the state. 

    Parameters
    ----------
    actor : model.DeterministicActor
        Deterministic actor object.
    state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
        State of the environment.

    Returns
    -------
    action : numpy.ndarray
        Action sampled from the actor's policy.

    """
    
    with torch.no_grad():
        action = actor.action(state).cpu().numpy()
        
        # clip the action, just in case
        action = np.clip(action, -1, 1)
        
        return action
    
    
def soft_update(target_model, model, factor=0.01):
    """
    Function to soft update a target model, given an initial model.

    Parameters
    ----------
    target_model : torch.nn.Module
        Target network to update.
    model : torch.nn.Module
        Initial network.
    factor : number, optional
        Smoothing factor. The default is 0.01.

    Returns
    -------
    None.

    """
    with torch.no_grad():
        for target_params, params in zip(target_model.parameters(), model.parameters()):
            target_params.data.copy_(factor * params + (1-factor) * target_params.data)   
    

def convert_to_tensor(x,device='cpu'):
    """
    Convert data array to torch tensor.

    Parameters
    ----------
    x : numpy array, torch.Tensor, torch.cuda.FloatTensor
        Data array for conversion.
    device : char, optional
        'cpu' or 'cuda'. The default is 'cpu'.

    Raises
    ------
    TypeError
        Type mismatch error.

    Returns
    -------
    None.

    """
    
    if isinstance(x, np.ndarray):
        xt = torch.from_numpy(x).float().to(device)
        
    elif isinstance(x, torch.Tensor) or isinstance(x, torch.cuda.FloatTensor):
        xt = x.to(device)
        
    else:
        raise TypeError("Input must be a numpy array or torch Tensor.")
        
    return xt

