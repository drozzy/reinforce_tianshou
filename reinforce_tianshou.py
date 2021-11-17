#%% Implementation of my reinforce in tianshou framework
# -Andriy Drozdyuk
import gym, torch, tianshou as ts
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.trainer.onpolicy import onpolicy_trainer

MAX_EPOCH = 50
STEPS_PER_EPOCH = 1_000
BATCH_SIZE = 1
LR = 0.005
γ = 0.9999
MAX_BUFFER_SIZE = 20_000 # This is max size of the buffer. In my case it only needs to be as long as a single episode
                         # as I clear the buffer after every episode

env = gym.make('CartPole-v0') # will be wrapped in DummyVecEnv automatically

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape, hidden_sizes=[64])
optim = torch.optim.Adam(net.parameters(), lr=LR)


class Reinforce(ts.policy.BasePolicy):
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
        self.optim.step()

        print(f'Reward={sum(rew)}')        
        return {'loss': loss.item()}

policy = Reinforce(net, optim, state_shape, action_shape)
train_collector = ts.data.Collector(policy, env, ts.data.ReplayBuffer(MAX_BUFFER_SIZE))
test_collector = ts.data.Collector(policy, env, ts.data.ReplayBuffer(MAX_BUFFER_SIZE))

result = onpolicy_trainer(policy, train_collector, test_collector, 
    max_epoch=MAX_EPOCH, step_per_epoch=STEPS_PER_EPOCH,
    repeat_per_collect=1, episode_per_test=10,
    episode_per_collect=1, batch_size=BATCH_SIZE)
    
FPS = 1/60.0
train_collector.collect(n_episode=5, render=FPS)
# %%
