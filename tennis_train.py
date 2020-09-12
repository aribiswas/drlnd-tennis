# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from utils import OUNoise, ExperienceBuffer, get_action, soft_update
from model import DeterministicActor, CentralizedCritic
from matplotlib import pyplot as plt
import numpy as np
import collections
import torch
import torch.optim as optim

# DDPG hyperparameters
BUFFER_LENGTH = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
ALPHA_CRITIC = 1e-3
ALPHA_ACTOR = 1e-4
TAU = 0.001
UPDATE_FREQ = 10

# training options
MAX_EPISODES = 5000      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0

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



# create environment
env = UnityEnvironment(file_name='tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# global experience buffer
buffer = ExperienceBuffer(osize, asize, num_agents, max_len=BUFFER_LENGTH)


# noise model
noise_model = OUNoise(size=[num_agents,asize], mean=0, mac=0.2, var=0.05, varmin=0.001, decay=5e-6, seed=seed)


# initialize actors and critics
actors = []
critics = []
target_actors = []
target_critics = []
actor_optimizers = []
critic_optimizers = []

for i in range(num_agents):
    # actor
    actors.append(DeterministicActor(osize, asize, seed).to(device))
    target_actors.append(DeterministicActor(osize, asize, seed))
    target_actors[i].load_state_dict(actors[i].state_dict())
    actor_optimizers.append(optim.Adam(actors[i].parameters(), lr=ALPHA_ACTOR))
    # critic
    critics.append(CentralizedCritic(osize * num_agents, asize, asize, seed).to(device))
    target_critics.append(CentralizedCritic(osize * num_agents, asize, asize, seed))
    target_critics[i].load_state_dict(critics[i].state_dict())
    critic_optimizers.append(optim.Adam(critics[i].parameters(), lr=ALPHA_CRITIC))



# ------  Train loop -------

for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    ep_reward = np.zeros(num_agents)
    
    while True:
        
        # step the noise model
        noise = noise_model.step()
        noise_log.append(np.mean(noise_model.x))
        
        # sample action from the current policy
        act1 = get_action(actors[0], states[0], noise[0], train=True)
        act2 = get_action(actors[1], states[1], noise[1], train=True)
        actions = np.stack([act1,act2]).reshape([num_agents,asize])
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = np.array(env_info.rewards)
        dones = np.array([1 if d is True else 0 for d in env_info.local_done])
        
        # store experience in buffer
        buffer.add(states.flatten(), 
                   actions.flatten(), 
                   rewards, 
                   next_states.flatten(), 
                   dones)
        
        states = next_states
        ep_reward += rewards
        step_count += 1
        
        aloss = np.zeros(num_agents)
        closs = np.zeros(num_agents)
        
        
        # ----  train actor and critic networks  ----
        
        if buffer.__len__() > BATCH_SIZE:
        
            # train actor and critic for each agent
            for i in range(num_agents):
                
                # create mini batch for training
                S, A, R, NS, D = buffer.sample(BATCH_SIZE)
                
                # compute td targets
                with torch.no_grad():
                    # target actions
                    TA1 = target_actors[0].mu(NS[:,:osize])
                    TA2 = target_actors[1].mu(NS[:,osize:])
                    # target critic value
                    tQ = target_critics[i].Q(NS,TA1,TA2).reshape(BATCH_SIZE)
                    # target value
                    Y = R[:,i] + GAMMA * tQ * (1-D[:,i])
                
                # compute local Q values
                A1 = A[:,:asize]
                A2 = A[:,asize:]
                Q = critics[i].Q(S,A1,A2).reshape(BATCH_SIZE)
                
                # critic loss
                critic_loss = torch.mean((Y-Q)**2)
                
                # update critic network
                critic_optimizers[i].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 1)  # gradient clipping
                critic_optimizers[i].step()
                
                # freeze critic before policy loss computation
                for p in critics[i].parameters():
                    p.requires_grad = False
                
                # actor loss
                ACT1 = actors[0].mu(S[:,:osize])
                ACT2 = actors[1].mu(S[:,osize:])
                actor_loss = -critics[i].Q(S,ACT1,ACT2).reshape(BATCH_SIZE).mean()
                
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
            for i in range(num_agents):
                if step_count % UPDATE_FREQ == 0:
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
    if VERBOSE and (ep_count==1 or ep_count%500==0):
        print('Episode: {:4d} \tAverage Reward: {:6.2f} \tActor Loss: [{:8.4f},{:8.4f}] \tCritic Loss: [{:8.4f},{:8.4f}] \tNoise: {:6.4f}'.format(ep_count,avg_reward,*(actor_loss_log[-1]),*(critic_loss_log[-1]),noise_log[-1]))
        
    # check if env is solved
    if not solved and avg_reward >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:6.2f}'.format(ep_count, avg_reward))
        solved = True

# save the policies
torch.save(actors[0].state_dict(), 'checkpoint_actor0.pth')
torch.save(actors[1].state_dict(), 'checkpoint_actor1.pth')

# Close environment
env.close()

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
    
    
    
    
    
    
    
    
    
    
    