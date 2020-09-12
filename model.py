# -*- coding: utf-8 -*-

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Actor class representing a deterministic policy mu(s). 
"""
class DeterministicActor(nn.Module):
    
    def __init__(self, num_obs, num_act, seed=0):
        
        torch.manual_seed(seed)

        super(DeterministicActor, self).__init__()

        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,num_act)
        self.tanh = nn.Tanh()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        
    def forward(self, state):
        """
        Perform forward pass through the network.
        
        """
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x
        
        
    def mu(self, state):
        """
        Get the deterministic policy.
        """
        
        return torch.clamp(self.forward(state), -1, 1)
    

"""
Critic class for approximating the state-action value function Q(s,a). 
"""
class CentralizedCritic(nn.Module):

    def __init__(self, num_obs, num_act1, num_act2, seed=0):
        
        torch.manual_seed(seed)

        super(CentralizedCritic, self).__init__()

        self.num_obs = num_obs

        # ------ layers ------
        
        # state path
        self.sfc1 = nn.Linear(num_obs,64)
        self.sfc2 = nn.Linear(64,64)
        
        # action1 path
        self.a1fc1 = nn.Linear(num_act1,64)
        self.a1fc2 = nn.Linear(64,64)
        
        # action2 path
        self.a2fc1 = nn.Linear(num_act2,64)
        self.a2fc2 = nn.Linear(64,64)
        
        # common path
        self.cfc1 = nn.Linear(64*3,64)
        self.cfc2 = nn.Linear(64,32)
        self.cfc3 = nn.Linear(32,1)

        
    def forward(self, state, action1, action2):
        """
        Perform forward pass through the network.
        
        """
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            s = torch.from_numpy(state).float().to(device)
        elif isinstance(state, torch.Tensor):
            s = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action1, numpy.ndarray):
            a1 = torch.from_numpy(action1).float().to(device)
        elif isinstance(action1, torch.Tensor):
            a1 = action1
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action2, numpy.ndarray):
            a2 = torch.from_numpy(action2).float().to(device)
        elif isinstance(action2, torch.Tensor):
            a2 = action2
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

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
        xc = F.relu(self.cfc1(xc))
        xc = F.relu(self.cfc2(xc))
        xc = self.cfc3(xc)

        return xc


    def Q(self, state, action1, action2):
        """
        Compute Q(s,a)
        """
        
        return self.forward(state, action1, action2)