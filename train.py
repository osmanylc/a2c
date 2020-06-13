import torch
import gym
import numpy as np
from tqdm import tqdm

from actor_critic import ActorCritic
from loss_function import value_loss, advantage_function, policy_loss
from util import sample_act, obs_to_tensor

def create_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_step(env, model, optimizer, c, discount, n_episodes):
    # Initialize loss and zero out gradients on parameters
    loss = 0
    optimizer.zero_grad()
    
    # Collect trajectories for n_episodes
    for _ in range(n_episodes):
        trajectory = []
        
        x_t = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                x_t = obs_to_tensor(x_t)
                _, log_pi_xt = model(x_t)
            
            a_t = sample_act(log_pi_xt)
            x_tp1, r_t, done, _ = env.step(a_t)
            
            trajectory.append((x_t, a_t, r_t))
            x_t = x_tp1
            
            # Note: We can collect the log_pi at every
            # step here, and the value at every state, 
            # so that we can combined them into the 
            # loss after we're done
        
        val_loss = value_loss(trajectory, discount, model)
        advantage = advantage_function(trajectory, discount, model)
        pi_loss = policy_loss(trajectory, advantage, model)
        
        loss += (pi_loss + c*val_loss)
        
        
    # Perform gradient step
    loss /= n_episodes
    loss.backward()
    optimizer.step()
    
    return loss


def train(
    n_epochs, 
    n_episodes,
    env_name='CartPole-v0',
    print_freq=100,
    discount=.999,
    lr=.01,
    c=.005
):
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    
    model = ActorCritic(in_size, out_size)
    optimizer = create_optimizer(model, lr)
    
    loss = 0
    for t in tqdm(range(n_epochs)):
        loss += train_step(env, model, optimizer, c, discount, n_episodes)
        
        if (t + 1) % print_freq == 0:
            print(loss / print_freq)
            loss = 0
        
    return model
