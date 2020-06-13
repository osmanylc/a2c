from random import random

import gym
import torch
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Linear


class ActorCritic(Module):
    
    def __init__(self, in_size, out_size, num_layers=2, num_hidden=64):
        super(ActorCritic, self).__init__()
        self.fc = [Linear(in_size, num_hidden)]
        
        for _ in range(num_layers - 1):
            self.fc.append(Linear(num_hidden, num_hidden))
        
        self.val = Linear(num_hidden, 1)
        self.pi = Linear(num_hidden, out_size)
        
    def forward(self, x):
        for layer in self.fc:
            x = F.relu(layer(x))
            
        val = self.val(x)
        log_pi = F.log_softmax(self.pi(x), dim=0)
        
        return val, log_pi

