import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from itertools import chain
from ..blocks.seqmlp import MLP


class Actor(nn.Module):
    def __init__(self, preprocess_net, pre_out, action_shape, *args, **kwargs):
        super(Actor, self).__init__()
        self.preprocess_net = preprocess_net
        self.actor_layer = nn.Linear(in_features=pre_out, out_features=action_shape)
        self.init_weight()        

    def forward(self, x):
        out = self.preprocess_net(x)
        out = self.actor_layer(out)
        logit = F.softmax(out, dim=-1)
        action_dist = dist.Categorical(logit)
        action = action_dist.sample()
        return action, logit

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0.01)


class Critic(nn.Module):
    def __init__(self, preprocess_net, pre_out, critic_value, *args, **Kwargs):
        super(Critic, self).__init__()
        self.preprocess_net = preprocess_net
        self.cirtic_layer = nn.Linear(in_features=pre_out, out_features=critic_value)
        self.init_weight()
    
    def forward(self, x):
        out = self.preprocess_net(x)
        out = self.cirtic_layer(out)
        return out
    
    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0.01)


