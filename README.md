# Advantage Actor Critic Implementation

![Cartpole agent.](cartpole.gif)

An implementation of the advantage actor critic algorithm to solve reinforcement learning problems. 

The algorithm uses a two-headed neural network, with one head computing the agent's policy and the other estimating the value of the given state. 

It is an online algorithm, using its current policy at each training step to collect a small batch of experience and use it to optimize the neural network parameters. The policy gradient and value function loss are calculated according to the formulas below:

<img src="https://render.githubusercontent.com/render/math?math=\hat{g} = \sum_{t=0}^{T-1} \log \pi(a_t | s_t, \theta) \left(\sum_{t' = t}^{T-1} r_{t'} - V_{\phi_k}(s_t)\right)">  .

<img src="https://render.githubusercontent.com/render/math?math=\hat{v} = \frac{1}{N} \sum_{1}^{N} (V_{\phi_k}(s) - \sum_{t' = t}^{T-1} r_{t'})^2">

The code can be run by calling the following command with Python 3 and PyTorch installed:

```bash
python evaluate.py
```

