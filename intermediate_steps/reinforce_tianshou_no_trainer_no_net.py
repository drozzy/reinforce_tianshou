#%% Implementation of my reinforce in tianshou framework
# Uses only the following parts from Tianshou framework:
# 1. Policy
# 2. Collector
#
# -Andriy Drozdyuk
import gym, torch, numpy as np, tianshou as ts
from typing import Any, Dict, Union, Callable, Optional
from torch import nn
from tianshou.data import Batch, Collector
from tianshou.policy import BasePolicy

MAX_STEPS = 50_000
BATCH_SIZE = 1
LR = 0.005
γ = 0.9999
MAX_BUFFER_SIZE = 20_000 # This is max size of the buffer. In my case it only needs to be as long as a single episode
                         # as I clear the buffer after every episode

env = gym.make('CartPole-v0') # will be wrapped in DummyVecEnv automatically

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(np.prod(state_shape), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, np.prod(action_shape)))

    def forward(self, obs, state=None, info={}):
        obs = torch.tensor(obs, dtype=torch.float).flatten(start_dim=1)

        return self.nn(obs), state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=LR)

class Reinforce(BasePolicy):
    def __init__(self, nn, optim, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.nn = nn
        self.optim = optim
        
    def forward(self, batch: Batch, state=None, **kwargs):    
        obs = batch.obs
        logits, _ = self.nn(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        return Batch(act=action, state=None, logits=logits)

    def learn(self, batch, batch_size, repeat):
        act, rew = batch.act, batch.rew

        result = self(batch)
        logits = result.logits
        
        dist = torch.distributions.Categorical(logits=logits)
        act = torch.tensor(act, dtype=torch.float)
        EligibilityVector = dist.log_prob(act)
        
        DiscountedReturns = []
        for t in range(len(rew)):
            G = 0.0
            for k, r in enumerate(rew[t:]):
                G += (γ**k)*r
            DiscountedReturns.append(G)

        DiscountedReturns = torch.tensor(DiscountedReturns, dtype=torch.float)
        loss = - torch.dot(EligibilityVector, DiscountedReturns)

        self.optim.zero_grad()
        loss.backward()
        optim.step()

        print(f'Reward={sum(rew)}')        
        return {'loss': loss.item()}

policy = Reinforce(net, optim, state_shape, action_shape)
collector = ts.data.Collector(policy, env, 
    ts.data.VectorReplayBuffer(MAX_BUFFER_SIZE, buffer_num=1))

def train(policy, collector:Collector, batch_size):   
    collector.reset_stat()
    
    step = 0

    while step <= MAX_STEPS:
        policy.train() # Switch nn mode to train
      
        result = collector.collect(n_episode=1)
        
        step += int(result["n/st"])
        
        reward = result['rews'][0] # Since we only have one episode
        
        USE_ALL_BUFFER_DATA = 0
        losses = policy.update(USE_ALL_BUFFER_DATA, collector.buffer, batch_size=batch_size, repeat=1)
        
        collector.reset_buffer()
        print(f'Step: {step}: Reward: {reward} ')

train(policy, collector, BATCH_SIZE)

FPS = 1/60.0
collector.collect(n_episode=5, render=FPS)
# %%
