# Reinforce Tianshou

My implementation of REINFORCE RL algorithm with the help of [tianshou](https://github.com/thu-ml/tianshou) reinforcement learning framework.
This builds on top of my implementation using only PyTorch here: https://github.com/drozzy/reinforce

The code trains an agent to solve a `CartPole-v0` environment and then renders a few episodes with a trained agent.

## Contents:

1. The `reinforce_tianshou.py` is the complete example.
1. The `intermediate_steps/reinforce_tianshou_no_trainer.py` shows how things would look without a trainer.
2. The `intermediate_steps/reinforce_tianshou_no_trainer_no_net.py`shows things without a trainer and a built-in network.
3. The `intermediate_steps/reinforce_tianshou_no_net.py` shows how to create a custom network and a custom policy, while using built-in trainer.
4. The `slides_code/policy_component.py` - shows an example of calling a built-in policy on an observation from CartPole environment.

## Install


    conda env create -f environment.yml
    conda activate reinforce_tianshou
    pip install -r requirements.txt

## Run

    python reinforce_tianshou.py

## References

- p.328 of Reinforcement Learning 2nd Ed. Sutton & Barto.
- Reinforce with Pytorch https://github.com/drozzy/reinforce

--Andriy Drozdyuk
