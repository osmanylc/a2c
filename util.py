from random import random
import torch


def obs_to_tensor(obs):
    obs = obs.astype('float32')
    obs = torch.from_numpy(obs)
    
    return obs


def sample_act(log_probs):
    u = random()
    p = torch.exp(log_probs)
    
    cum_p = 0
    
    for i in range(len(log_probs)):
        cum_p += p[i]
        
        if u <= cum_p:
            return i
    
    # Return last action in case there is a
    # rounding error and cum_p doesn't go to 1
    return len(log_probs)
