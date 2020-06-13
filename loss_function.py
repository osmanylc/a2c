import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Linear


def policy_loss(trajectory, advantage, model):
    """
    A trajectory is a sequence of the following:
    
    (x_t, a_t, r_t)
    
    x_t: The observation at time t
    a_t: The action taken at time t, given x_t.
    r_t: The reward received for taking action a_t at x_t.
    """
    n = len(trajectory)
    pi_loss = 0
    
    for t in range(n):
        adv_t = advantage[t]
        x_t, a_t, r_t = trajectory[t]
        _, log_pi_xt = model(x_t)
        
        pi_loss -= log_pi_xt[a_t] * adv_t
        
    pi_loss /= n
    
    return pi_loss


def value_loss(trajectory, discount, model):
    n = len(trajectory)
    rw_to_go = trajectory_return(trajectory, discount)
    ret = torch.empty((n,1))
    
    for t in range(n):
        ret[t] = rw_to_go[t]
    
    # Calculate advantage values with gradient
    xs, _, _ = zip(*trajectory)
    x_tensor = torch.stack(xs)
    vals, _ = model(x_tensor)
    
    return F.mse_loss(vals, ret)


def trajectory_return(trajectory, discount):
    n = len(trajectory)
    rw_to_go = {}
    rw_sum = 0
    
    # Calculate suffix-sums of reward
    for t in reversed(range(n)):
        _, _, r_t = trajectory[t]
        rw_sum = r_t + discount * rw_sum
        rw_to_go[t] = rw_sum
        
    return rw_to_go


def advantage_function(trajectory, discount, model):
    """
    Create a dictionary with the advantage at each
    timestep.
    """
    n = len(trajectory)
    advantage = []
    rw_to_go = {}
    rw_sum = 0
    
    # Calculate suffix-sums of reward
    for t in reversed(range(n)):
        _, _, r_t = trajectory[t]
        rw_sum = r_t + discount * rw_sum
        rw_to_go[t] = rw_sum
        
    # Calculate advantage
    for t in range(n):
        x_t, _, _ = trajectory[t]
        
        with torch.no_grad():
            val_xt, _ = model(x_t)
            advantage.append(rw_to_go[t] - val_xt)
    
    return torch.stack(advantage)
