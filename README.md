# Reinforce Tianshou

My implementation of REINFORCE RL algorithm with the help of [tianshou](https://github.com/thu-ml/tianshou) reinforcement learning framework.
This builds on top of my implementation using only PyTorch here: https://github.com/drozzy/reinforce

The `reinforce_tianshou.py` script trains an agent to solve a `CartPole-v0` environment and then renders a few episodes with a trained agent.

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
