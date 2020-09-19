# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import convert_to_tensor

def hidden_init(layer):
    """
    Initialize hidden layer.

    Parameters
    ----------
    layer : torch.nn layer
        Hidden layer.

    Returns
    -------
    TYPE
        Tuple.
    lim : number
        Scalar limit.

    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class DeterministicActor(nn.Module):
    """
    Actor class representing a deterministic policy. 

    Attributes
    ----------
    num_obs : number
        Size of states.
    num_act : number
        Size of actions.

    Methods
    -------
    reset_parameters():
        Resets the weights of the layers.
    forward(state):
        Perform forward pass through the network.
    action(state):
        Sample action from the policy, given the state.
        
    """
    
    def __init__(self, num_obs, num_act, seed=0):
        """
        Initialize actor network.

        Parameters
        ----------
        num_obs : number
            Size of observations.
        num_act : number
            Size of actions.
        seed : number, optional
            Random seed. The default is 0.

        Returns
        -------
        None.

        """
        
        torch.manual_seed(seed)

        super(DeterministicActor, self).__init__()

        self.num_obs = num_obs
        self.num_act = num_act

        # layers
        self.fc1    = nn.Linear(num_obs,256)
        self.fc2    = nn.Linear(256,128)
        self.fc3    = nn.Linear(128,num_act)
        self.tanh   = nn.Tanh()
        self.reset_parameters()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def reset_parameters(self):
        """
        Reset network weights.

        Returns
        -------
        None.

        """
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """
        Perform forward pass through the network.

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            State of the environment.

        Returns
        -------
        x : torch.Tensor, torch.cuda.FloatTensor
            Network output.

        """
        
        # convert to torch
        x = convert_to_tensor(state, self.device)
        
        # forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # restrict action from -1 to 1 using tanh
        
        return x
        
        
    def action(self, state):
        """
        Sample action from the policy, given the state.

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.Float
            State of the environment.
        
        Returns
        -------
        torch.Tensor, torch.cuda.Float
            Sampled action.

        """
        
        return self.forward(state)
    


class CentralizedCritic(nn.Module):
    """
    Critic class for approximating the state-action value function Q(s,a1,a2).

    Attributes
    ----------
    num_states : number
        Size of states.
    num_act1 : number
        Size of action of first agent.
    num_act2 : number
        Size of action of second agent.

    Methods
    -------
    forward(state,action1,action2):
        Perform forward pass through the network.
    Q(state,action1,action2):
        Compute the state-action value Q(s,a1,a2).
        
    """

    def __init__(self, num_states, num_act1, num_act2, seed=0):
        """
        Initialize critic network.

        Parameters
        ----------
        num_states : number
            Size of states.
        num_act1 : number
            Size of action of first agent..
        num_act2 : number
            Size of action of second agent..
        seed : number, optional
            Random seed. The default is 0.

        Returns
        -------
        None.

        """
        
        torch.manual_seed(seed)

        super(CentralizedCritic, self).__init__()

        self.num_states = num_states

        # ------ layers ------
        
        # state path
        self.sfc1 = nn.Linear(num_states,256)
        self.sfc2 = nn.Linear(256,64)
        
        # action1 path
        self.a1fc1 = nn.Linear(num_act1,64)
        self.a1fc2 = nn.Linear(64,64)
        
        # action2 path
        self.a2fc1 = nn.Linear(num_act2,64)
        self.a2fc2 = nn.Linear(64,64)
        
        # common path
        self.cbn1 = nn.BatchNorm1d(64*3)
        self.dro1 = nn.nn.Dropout(p=0.2)
        self.cfc1 = nn.Linear(64*3,64)
        self.cfc2 = nn.Linear(64,64)
        self.cfc3 = nn.Linear(64,1)

        
    def forward(self, state, action1, action2):
        """
        Perform forward pass through the network.

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            State of the environment.
        action1 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of first agent.
        action2 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of second agent.

        Returns
        -------
        xc : torch.Tensor, torch.cuda.FloatTensor
            Network output.

        """
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # convert to torch
        s = convert_to_tensor(state,device)
        a1 = convert_to_tensor(action1,device)
        a2 = convert_to_tensor(action2,device)
            
        isbatch = True if len(s.size())>1 else False

        # state path
        xs = F.relu(self.sfc1(s))
        xs = F.relu(self.sfc2(xs))
        
        # action1 path
        xa1 = F.relu(self.a1fc1(a1))
        xa1 = F.relu(self.a1fc2(xa1))
        
        # action2 path
        xa2 = F.relu(self.a2fc1(a2))
        xa2 = F.relu(self.a2fc2(xa2))
        
        # common path
        xc = torch.cat((xs,xa1,xa2), dim=1)
        # apply batch norm for batch inputs
        if isbatch:
            xc = self.cbn1(xc)
        xc = F.relu(self.cfc1(xc))
        xc = F.relu(self.cfc2(xc))
        xc = self.dro1(xc)
        xc = self.cfc3(xc)

        return xc


    def Q(self, state, action1, action2):
        """
        Compute the state-action value Q(s,a1,a2).

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            State of the environment.
        action1 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of first agent.
        action2 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of second agent.

        Returns
        -------
        torch.Tensor, torch.cuda.FloatTensor
            State-action value Q(s,a).

        """
        
        return self.forward(state, action1, action2)
    
    

class QCritic(nn.Module):
    """
    Critic class for approximating the state-action value function Q(s,a).

    Attributes
    ----------
    num_obs : number
        Size of states.
    num_act : number
        Size of actions.

    Methods
    -------
    forward(state):
        Perform forward pass through the network.
    action(state):
        Sample action from the policy, given the state.
        
    """

    def __init__(self, num_obs, num_act, seed=0):
        """
        Initialize critic network.

        Parameters
        ----------
        num_obs : number
            Size of states.
        num_act : number
            Size of action.
        seed : number, optional
            Random seed. The default is 0.

        Returns
        -------
        None.

        """

        super(QCritic, self).__init__()

        torch.manual_seed(seed)
        self.num_obs = num_obs

        # ------ layers ------
        
        # state path
        self.sfc1 = nn.Linear(num_obs,256)
        self.cfc1 = nn.Linear(256+num_act,128)
        self.cfc2 = nn.Linear(128,1)
        self.reset_parameters()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        
    def reset_parameters(self): 
        """
        Reset network weights.

        Returns
        -------
        None.

        """
        
        self.sfc1.weight.data.uniform_(*hidden_init(self.sfc1))
        self.cfc1.weight.data.uniform_(*hidden_init(self.cfc1))
        self.cfc2.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, state, action):
        """
        Perform forward pass through the network.

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            State of the environment.
        action : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of the agent.

        Returns
        -------
        xc : torch.Tensor, torch.cuda.FloatTensor
            Network output.

        """
        
        # convert to torch
        s = convert_to_tensor(state,self.device)
        a = convert_to_tensor(action,self.device)

        xs = F.relu(self.sfc1(s))
        xc = torch.cat((xs,a), dim=1)
        xc = F.relu(self.cfc1(xc))
        xc = self.cfc2(xc)

        return xc


    def Q(self, state, action):
        """
        Compute the state-action value Q(s,a1,a2).

        Parameters
        ----------
        state : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            State of the environment.
        action1 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of first agent.
        action2 : numpy.ndarray, torch.Tensor, torch.cuda.FloatTensor
            Action of second agent.

        Returns
        -------
        torch.Tensor, torch.cuda.FloatTensor
            State-action value Q(s,a).

        """
        
        return self.forward(state, action)
    
    
    
    
    
    
    