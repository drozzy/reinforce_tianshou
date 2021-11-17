
#%% Slides for the policy component

import gym, torch, tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.policy.modelfree.pg import PGPolicy
LR = 0.005
γ = 0.9999
env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape, hidden_sizes=[64])
optim = torch.optim.Adam(net.parameters(), lr=LR)
dist = torch.distributions.categorical.Categorical
policy = PGPolicy(net, optim, dist, γ)

# %%
policy
# %%
obs = env.reset()
obs
#%%
from tianshou.data import Batch
batch = Batch(obs=[obs])
out = policy(batch)
out
# %%
out.dist.probs
# %%
out.dist.sample()
# %%

# %%