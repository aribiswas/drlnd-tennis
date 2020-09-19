# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from utils import OUNoise, ExperienceBuffer, action_with_noise, soft_update
from model import DeterministicActor, CentralizedCritic
from matplotlib import pyplot as plt
import numpy as np
import collections
import copy
import torch
import torch.optim as optim

# DDPG hyperparameters
BUFFER_LENGTH = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
ALPHA_CRITIC = 1e-3
ALPHA_ACTOR = 1e-3
TAU = 0.01
UPDATE_FREQ = 10

# training options
MAX_EPISODES = 5000      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# observation and action specs of each agent
osize = 24
asize = 2
num_agents = 2

# score logs
reward_log = []
avg_log = []
avg_window = collections.deque(maxlen=AVG_WINDOW)
actor_loss_log = [[0,0]]
critic_loss_log = [[0,0]]
noise_log = [0]

# verbosity
VERBOSE = True
solved = False
step_count = 0


# ----- Setup -----

# create environment
env = UnityEnvironment(file_name='tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# create experience buffer
buffer = ExperienceBuffer(osize, asize, num_agents, max_len=BUFFER_LENGTH)


# create noise models
np.random.seed(0)   # set the numpy seed
noise_model_0 = OUNoise(size=asize, mean=0, mac=0.12, var=0.2, varmin=0.01, decay=1e-4)
noise_model_1 = OUNoise(size=asize, mean=0, mac=0.12, var=0.2, varmin=0.01, decay=1e-4)

# create actors and critics
actors = []
critics = []
target_actors = []
target_critics = []
actor_optimizers = []
critic_optimizers = []

for i in range(num_agents):
    # actor
    actors.append(DeterministicActor(osize, asize, seed=0).to(device))
    target_actors.append(DeterministicActor(osize, asize, seed=0).to(device))
    target_actors[i].load_state_dict(actors[i].state_dict())
    actor_optimizers.append(optim.Adam(actors[i].parameters(), lr=ALPHA_ACTOR))
    # critic
    critics.append(CentralizedCritic(osize * num_agents, asize, asize, seed=0).to(device))
    target_critics.append(CentralizedCritic(osize * num_agents, asize, asize, seed=0).to(device))
    target_critics[i].load_state_dict(critics[i].state_dict())
    critic_optimizers.append(optim.Adam(critics[i].parameters(), lr=ALPHA_CRITIC))


# ------  Train loop -------

for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    ep_reward = np.zeros(num_agents)
    ep_steps = 1
    
    while True:
        
        # step the noise model
        noise_0 = noise_model_0.step()
        noise_1 = noise_model_1.step()
        noise_log.append([noise_model_0.var,noise_model_1.var])
        
        # sample action from the current policy
        act1 = action_with_noise(actors[0], states[0], noise_0, train=True)
        act2 = action_with_noise(actors[1], states[1], noise_1, train=True)
        actions = np.stack([act1,act2]).reshape([num_agents,asize])
        #print(actions)
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = np.array(env_info.rewards)
        dones = np.array([1 if d is True else 0 for d in env_info.local_done])
        
        # store experience in buffer
        # flatten states and actions before storing. The states/actions will be
        # stacked horizontally in a nx1 array where
        # n = num_agents * osize for states
        #   = num_agents * asize for actions
        buffer.add(states.flatten(), 
                   actions.flatten(), 
                   rewards, 
                   next_states.flatten(), 
                   dones)
        
        states = next_states
        ep_reward += rewards
        step_count += 1
        ep_steps += 1
        
        aloss = np.zeros(num_agents)
        closs = np.zeros(num_agents)
        
        
        # ----  train actor and critic networks  ----
        
        if buffer.__len__() > BATCH_SIZE:
        
            # train actor and critic for each agent
            for i in range(num_agents):
                
                # create mini batch for training
                # S: states, A: actions, R: rewards, NS: next states, D: dones
                S, A, R, NS, D = buffer.sample(BATCH_SIZE)
                
                # ----- train critic -----
                
                # compute td targets
                with torch.no_grad():
                    # next observations
                    NO1 = NS[:,:osize]
                    NO2 = NS[:,osize:]
                    # target actions (decentralized execution)
                    TA1 = target_actors[0].action(NO1) 
                    TA2 = target_actors[1].action(NO2)
                    # target critic value
                    tQ = target_critics[i].Q(NS,TA1,TA2)
                    # target value
                    # indexing a torch variable R[:,i:i+1] preserves shape in
                    # contrast to R[:,i] which does not
                    Y = R[:,i:i+1] + GAMMA * tQ * (1-D[:,i:i+1])
                
                # compute local Q values
                A1 = A[:,:asize]
                A2 = A[:,asize:]
                Q = critics[i].Q(S,A1,A2)
                
                # critic loss
                critic_loss = torch.mean((Y-Q).pow(2))
                
                # update critic network
                critic_optimizers[i].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 1)  # gradient clipping
                critic_optimizers[i].step()
                
                # ----- train actor -----
                
                # freeze critic before policy loss computation
                for p in critics[i].parameters():
                    p.requires_grad = False
                
                # compute actor loss
                # dont use A here, compute actions again. These will have grad
                O1 = S[:,:osize]
                O2 = S[:,osize:]
                A1 = actors[0].action(O1)   # get action from current policy
                A2 = actors[1].action(O2)
                actor_loss = -critics[i].Q(S,A1,A2).mean()  # -ve sign for gradient ascent
                
                # update actor network
                actor_optimizers[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), 1)  # gradient clipping
                actor_optimizers[i].step()
                
                # Unfreeze critic
                for p in critics[i].parameters():
                    p.requires_grad = True
                
                aloss[i] = actor_loss.detach().cpu().numpy()
                closs[i] = critic_loss.detach().cpu().numpy()
                
            # soft update target actors and critics
            if step_count % UPDATE_FREQ == 0:
                for i in range(num_agents):
                    soft_update(target_actors[i], actors[i], factor=TAU)
                    soft_update(target_critics[i], critics[i], factor=TAU)
                    
            # log the losses
            actor_loss_log.append(aloss.tolist())
            critic_loss_log.append(closs.tolist())
        
        # terminate if done
        if np.any(dones):
            break
    
    # print training progress
    avg_window.append(np.max(ep_reward))
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%10==0):
        print('Ep: {:4d} | Steps: {:4d} | AvR: {:.4f} | ALoss: [{:.4e},{:.4e}] | CLoss: [{:.4e},{:.4e}] | Noise Var: [{:.4f},{:.4f}]'.format(ep_count,ep_steps,avg_reward,*(actor_loss_log[-1]),*(critic_loss_log[-1]),*(noise_log[-1])))
        
    # check if env is solved
    if not solved and avg_reward >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.4f}'.format(ep_count, avg_reward))
        solved = True

# save the policies
torch.save(actors[0].state_dict(), 'checkpoint_actor0.pth')
torch.save(actors[1].state_dict(), 'checkpoint_actor1.pth')

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
ax2.plot(actor_loss_log)

ax3 = axarr[1]
ax3.set_xlabel("Steps")
ax3.set_ylabel("Critic Loss")
ax3.plot(critic_loss_log)

fig2.tight_layout(pad=1.0)
plt.show()
fig1.savefig('results_ddpg_scores.png',dpi=200)
fig2.savefig('results_ddpg_losses.png',dpi=200)
    
    
    
    
    
    
    
    
    
    
    