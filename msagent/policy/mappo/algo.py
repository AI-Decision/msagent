import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import chain
import torch
from msagent.policy import MAPPOPolicy


def convert(input):
    if isinstance(input, np.ndarray):
        out = torch.from_numpy(input).to("cuda:0")
    elif isinstance(input, torch.Tensor):
        out = input
    else:
        raise NotImplementedError("the {} type can not be supported now".format(type(input)))
    return out


class MAPPOAlgo(object):
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.num_agents = args.num_agents
        self.policy = MAPPOPolicy(args, obs_shape=115, share_obs_shape = self.num_agents * 115, action_shape=19, device=device)
        self.use_max_grad_clip = args.use_max_grad_clip
        self.epoch = args.epoch
        self.coef = args.coef
        self.epsilon = args.epsilon
        self.use_adv_norm = args.use_adv_norm
        self.use_all_reduce = args.use_all_reduce
        self.tpdv = dict(dtype=torch.float32, device=device)

    def calcu_critic_loss(self, value, value_pred_batch, return_batch):
        value_pred_clipped = value_pred_batch + (value - value_pred_batch).clamp(-0.1, 0.1)
        error_clipped = return_batch - value_pred_clipped
        critic_loss = F.smooth_l1_loss(value, return_batch)
        return critic_loss

    def average_gradient(self, params_list):
        size = np.float64(torch.distributed.get_world_size())
        for param in params_list:
            # print(rank, param.grad.data, "\n")
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # print(rank, param.grad.data, "\n")
            param.grad.data /= size

    def update(self, batch):
        share_obs_batch, obs_batch, action_batch, value_pred_batch, return_batch, action_dist_batch, adv_batch = batch

        if self.use_adv_norm:
            mean_adv = np.mean(adv_batch)
            std_adv = np.mean(adv_batch)
            adv_batch = (adv_batch - mean_adv) / std_adv

        share_obs_batch = convert(share_obs_batch).to(**self.tpdv)
        obs_batch = convert(obs_batch).to(**self.tpdv)
        action_batch = convert(action_batch).to(**self.tpdv)
        value_pred_batch = convert(value_pred_batch).to(**self.tpdv)
        return_batch = convert(return_batch).to(**self.tpdv)
        action_dist_batch = convert(action_dist_batch).to(**self.tpdv)
        adv_batch = convert(adv_batch).to(**self.tpdv)

        _, new_action_dist_batch = self.policy.get_action_prob(obs_batch)
        new_value_batch = self.policy.get_value(share_obs_batch)

        old_dist = dist.Categorical(action_dist_batch)
        new_dist = dist.Categorical(new_action_dist_batch)

        old_logprob = old_dist.log_prob(action_batch)
        new_logprob = new_dist.log_prob(action_batch)

        ratio = torch.exp(new_logprob- old_logprob)

        actor_loss = -torch.mean(torch.min(ratio * adv_batch, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_batch))

        entropy_loss = torch.mean(new_dist.entropy())

        self.policy.actor_optimizer.zero_grad()
        (actor_loss - entropy_loss).backward()

        if self.use_max_grad_clip:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.use_max_grad_clip)
        
        self.policy.actor_optimizer.step()

        critic_loss = self.calcu_critic_loss(new_value_batch, value_pred_batch, return_batch)
        
        self.policy.critic_optimizer.zero_grad()
        
        (critic_loss * self.coef).backward()

        if self.use_max_grad_clip:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.use_max_grad_clip)

        if self.use_all_reduce:
            self.average_gradient(list(self.policy.actor.parameters()) + 
                                  list(self.policy.critic.parameters()))

        self.policy.critic_optimizer.step()

        return actor_loss, critic_loss, actor_grad_norm, critic_grad_norm

    def train(self, buffer):

        train_info = defaultdict(lambda: 0)

        for _ in range(self.epoch):
            data_generator = buffer.sample()
            for data in data_generator:
                actor_loss, critic_loss, actor_grad_norm, critic_grad_norm = self.update(data)
                # # can not convert CUDA tensor to numpy
                train_info['actor_loss'] += actor_loss.detach().cpu().numpy()
                train_info['critic_loss'] += critic_loss.detach().cpu().numpy()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
        
        for key in train_info:
            train_info[key] /= self.epoch

        return train_info
