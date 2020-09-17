# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from utils import OUNoise, ExperienceBuffer
from model import DeterministicActor, QCritic
from agents import DDPGAgent
from matplotlib import pyplot as plt
import numpy as np
import collections
import torch
import torch.optim as optim

# DDPG hyperparameters
params = {
    "BUFFER_LENGTH": int(1e5),
    "BATCH_SIZE": 128,
    "GAMMA": 0.995,
    "ACTOR_LR": 1e-4,
    "CRITIC_LR": 2e-4,
    "TAU": 0.05,
    "UPDATE_FREQ": 2,
    "TRAIN_ITERS": 10,
    "NOISE_SIZE": 2,
    "NOISE_MEAN": 0,
    "NOISE_MAC": 0.13,
    "NOISE_VAR": 0.2,
    "NOISE_VARMIN": 0.001,
    "NOISE_DECAY": 1e-5,
    }

# training options
MAX_EPISODES = 800      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages
TRAIN_EVERY = 5          # Episode interval for training
PRINT_EVERY = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# observation and action specs of each agent
osize = 24
asize = 2
num_agents = 2

# score logs
reward_log = []
avg_log = []
avg_window = collections.deque(maxlen=AVG_WINDOW)

# verbosity
VERBOSE = True
solved = False


# ----- Setup -----

# create environment
env = UnityEnvironment(file_name='tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# create experience buffer
buffer = ExperienceBuffer(osize, asize, max_len=params["BUFFER_LENGTH"])

# create noise models
np.random.seed(0)   # set the numpy seed

# create actor network
actor = DeterministicActor(osize, asize, seed=0).to(device)
target_actor = DeterministicActor(osize, asize, seed=0).to(device)

# create critic network
critic = QCritic(osize, asize, seed=0).to(device)
target_critic = QCritic(osize, asize, seed=0).to(device)


# create DDPG agents
agent_0 = DDPGAgent(actor, critic, target_actor, target_critic, buffer, params)
agent_1 = DDPGAgent(actor, critic, target_actor, target_critic, buffer, params)

# ------  Train loop -------

for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    ep_reward = np.zeros(num_agents)
    ep_steps = 1
    
    while True:
        
        # sample action from the current policy
        act_0 = agent_0.get_action(states[0], train=True)
        act_1 = agent_1.get_action(states[1], train=True)
        actions = np.concatenate((act_0, act_1), axis=0)
        actions = np.reshape(actions,(1,4))
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = np.array(env_info.rewards)
        dones = np.array([1 if d is True else 0 for d in env_info.local_done])
        
        # step the agents
        dotrain = True if ep_count%TRAIN_EVERY==0 else False
        agent_0.step(states[0],act_0,rewards[0],next_states[0],dones[0],dotrain)
        agent_1.step(states[1],act_1,rewards[1],next_states[1],dones[1],dotrain)
        
        states = next_states
        ep_reward += rewards
        ep_steps += 1
        
        # terminate if done
        if np.any(dones):
            break
    
    # print training progress
    avg_window.append(np.max(ep_reward))
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%PRINT_EVERY==0):
        print('Episode: {:4d} | Steps: {:4d} | Average Reward: {:.4f}'.format(ep_count,ep_steps,avg_reward))
        
    # check if env is solved
    if not solved and avg_reward >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.4f}'.format(ep_count, avg_reward))
        solved = True

# save the actor
torch.save(actor.state_dict(), 'checkpoint_actor.pth')

# Close environment
env.close()


# ----- Results -----

# plot score history
plt.ion()
fig1, ax1 = plt.subplots(1,1, figsize=(8,4), dpi=200)
ax1.set_title("Training Results")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average Reward")
ax1.plot(avg_log)

fig2, axarr = plt.subplots(2,1, figsize=(6,4), dpi=200)
# plot loss
ax2 = axarr[0]
ax2.set_xlabel("Steps")
ax2.set_ylabel("Actor Loss")
ax2.plot(agent_0.actor_loss_log)

ax3 = axarr[1]
ax3.set_xlabel("Steps")
ax3.set_ylabel("Critic Loss")
ax3.plot(agent_0.critic_loss_log)

fig2.tight_layout(pad=1.0)
plt.show()
fig1.savefig('results_ddpg_scores.png',dpi=200)
fig2.savefig('results_ddpg_losses.png',dpi=200)
    
    
    
    
    
    
    
    
    
    
    