import torch
import numpy as np
from ..blocks.seqmlp import MLP
from .network import Actor, Critic
import torch.nn as nn


def convert(input):
    if isinstance(input, np.ndarray):
        out = torch.from_numpy(input).to("cuda:0")
    elif isinstance(input, torch.Tensor):
        out = input
    else:
        raise NotImplementedError("the {} type can not be supported now".format(type(input)))
    return out


class MAPPOPolicy(nn.Module):

    def __init__(self, args, obs_shape, share_obs_shape, action_shape, device=torch.device('cpu')):
        
        self.lr = args.lr
        self.obs_shape = obs_shape
        self.share_obs_shape = share_obs_shape
        self.action_shape = action_shape

        self.actor_prenet = MLP(obs_shape, [128,256], 128)
        self.critic_prenet = MLP(share_obs_shape, [256], 64)
        self.device = device
        
        self.actor = Actor(self.actor_prenet, 128, action_shape).to(device)
        self.critic = Critic(self.critic_prenet, 64, 1).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    

    def get_action_prob(self, obs):
        action, logit = self.actor(convert(obs))
        return action, logit

    
    def get_value(self, obs):
        value = self.critic(convert(obs))
        return value
    
    @torch.no_grad()
    def eval_actor_prob(self, obs):
        action, logit = self.actor(convert(obs))
        return action, logit
    
    @torch.no_grad()
    def eval_critic_value(self, obs):
        value = self.critic(convert(obs))
        return value
    
    def forward(self):
        pass