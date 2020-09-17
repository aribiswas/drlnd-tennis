# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


"""
Actor class representing a deterministic policy mu(s). 
"""
class DeterministicActor(nn.Module):
    
    def __init__(self, num_obs, num_act, seed=0):
        
        torch.manual_seed(seed)

        super(DeterministicActor, self).__init__()

        self.num_obs = num_obs
        self.num_act = num_act

        # layers
        self.fc1 = nn.Linear(num_obs,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_act)
        self.tanh = nn.Tanh()
        self.reset_parameters()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """
        Perform forward pass through the network.
        
        """
        
        # convert to torch
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor) or isinstance(state, torch.cuda.FloatTensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # restrict action from -1 to 1 using tanh
        return x
        
        
    def mu(self, state):
        """
        Get the deterministic policy.
        """
        
        return self.forward(state)
    

"""
Critic class for approximating the state-action value function Q(s,a1,...an). 

"""
class CentralizedCritic(nn.Module):

    def __init__(self, num_states, num_act1, num_act2, seed=0):
        
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
        
        """
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # convert to torch
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state).float().to(device)
        elif isinstance(state, torch.Tensor) or isinstance(state, torch.cuda.FloatTensor):
            s = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action1, np.ndarray):
            a1 = torch.from_numpy(action1).float().to(device)
        elif isinstance(action1, torch.Tensor) or isinstance(action1, torch.cuda.FloatTensor):
            a1 = action1
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action2, np.ndarray):
            a2 = torch.from_numpy(action2).float().to(device)
        elif isinstance(action2, torch.Tensor) or isinstance(action2, torch.cuda.FloatTensor):
            a2 = action2
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
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
        Compute Q(s,a)
        """
        
        return self.forward(state, action1, action2)
    
    
"""
Critic class for approximating the state-action value function Q(s,a). 

"""
class QCritic(nn.Module):

    def __init__(self, num_obs, num_act, seed=0):
        """
        Initialize a Q-value critic network.
        
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

    def reset_parameters(self): 
        """
        Reset network weights
        
        """
        
        self.sfc1.weight.data.uniform_(*hidden_init(self.sfc1))
        self.cfc1.weight.data.uniform_(*hidden_init(self.cfc1))
        self.cfc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """
        Perform forward pass through the network.
        
        """
        
        # convert to torch
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            s = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action, np.ndarray):
            a = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            a = action
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        xs = F.relu(self.sfc1(s))
        xc = torch.cat((xs,a), dim=1)
        xc = F.relu(self.cfc1(xc))
        xc = self.cfc2(xc)

        return xc


    def Q(self, state, action):
        """
        Compute Q(s,a)
        """
        
        return self.forward(state, action)
    
    
    
    
    
    
    