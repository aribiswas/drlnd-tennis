# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from utils import sim_act
from model import DeterministicActor
import numpy as np
import torch

# sim options
NUM_SIMS = 5      # Maximum number of training episodes

# observation and action specs of each agent
osize = 24
asize = 2

# create environment
env = UnityEnvironment(file_name='tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# initialize actors and critics
actor = DeterministicActor(osize, asize, seed=0)
actor.load_state_dict(torch.load('checkpoint_actor_cpu.pth'))

# ------  Train loop -------

for ep_count in range(NUM_SIMS):

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    
    while True:
        
        # sample action from the current policy
        act1 = sim_act(actor, states[0])
        act2 = sim_act(actor, states[1])
        actions = np.concatenate((act1, act2), axis=0)
        actions = np.reshape(actions,(1,4))
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        dones = env_info.local_done
        states = next_states
        
        # terminate if done
        if np.any(dones):
            break
    
# Close environment
env.close()
    
    
    
    
    
    