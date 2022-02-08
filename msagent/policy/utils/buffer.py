
import numpy as np
import os
from collections import deque
import torch
import random
from segment_tree import SumSegmentTree, MinSegmentTree

class SingleAgentReplayBuffer(object):
    
    def __init__(self, obs_dim=115, action_dim=19, size=10000, batch_size=32):

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.v_buf = np.zeros([size, 1])
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, 1], dtype=np.float32)
        self.actprob_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
    

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts = self.acts_buf[idxs],
                    rews = self.rews_buf[idxs],
                    done = self.done_buf[idxs])


    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(SingleAgentReplayBuffer):
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6
    ):
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta  = 0.4):
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight



class MultiAgentReplayBuffer(object):

    def __init__(self, args, obs_shape=(115,), act_shape=19):
        self.episode_len = args.episode_length
        self.num_agents = args.num_agents
        self.obs_shape = obs_shape
        share_obs_shape = (self.num_agents * obs_shape[0],)
        self.act_shape = act_shape
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.use_gae = args.use_gae

        self.obs = np.zeros((self.episode_len, self.num_agents, obs_shape[0]), dtype=np.float32)
        self.share_obs = np.zeros((self.episode_len, self.num_agents, share_obs_shape[0]), dtype=np.float32)
        self.value_pred = np.zeros((self.episode_len, self.num_agents, 1),dtype=np.float32)
        self.returns = np.zeros_like(self.value_pred,dtype=np.float32)
        self.advantage = np.zeros_like(self.value_pred, dtype=np.float32)

        self.actions = np.zeros((self.episode_len, self.num_agents, 1),dtype=np.float32)
        self.rewards = np.zeros_like(self.actions, dtype=np.float32)
        
        self.dones = np.zeros_like(self.actions, dtype=np.float32)
        self.action_prob = np.zeros((self.episode_len, self.num_agents, act_shape),dtype=np.float32)
        self.step = 0 

    def insert(self, share_obs, obs, actions, action_prob, value_pred, reward, done):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.actions[self.step] = np.expand_dims(actions,axis=1)
        self.action_prob[self.step] = action_prob.copy()
        self.value_pred[self.step] = value_pred.copy()
        self.rewards[self.step] = np.expand_dims(reward,axis=1)
        self.dones[self.step] = np.expand_dims(done,axis=1)
        self.step = (self.step+1) % self.episode_len

    def compute_return(self):
        next_value = self.value_pred[-1,:,:]
        next_mask = self.dones[-1,:,:]
        if self.use_gae:
            gae = 0

            for step in reversed(range(self.actions.shape[0] - 1)):
                delta = self.rewards[step] + self.gamma * next_value * (1 - next_mask) - self.value_pred[step]
                gae = delta + self.gamma * self.gae_lambda * gae
                self.returns[step] = gae + self.value_pred[step]
                next_value = self.value_pred[step]
                next_mask = self.dones[step]
                self.advantage[step] = self.returns[step] - self.value_pred[step]
        else:
            for step in reversed(range(self.actions.shape[0])):
                self.returns[step] = next_value * self.gammma * (1 - next_mask) + self.rewards[step]
                next_value = self.returns[step]
                next_mask = self.dones[step]
                self.advantage[step] = self.returns[step] - self.value_pred[step]

    def sample(self, num_batch=2):
        indices = torch.randperm(self.episode_len-1).numpy()
        obs = self.obs.reshape(-1, self.obs.shape[-1])
        share_obs = self.share_obs.reshape(-1, self.share_obs.shape[-1])
        actions = self.actions.reshape(-1, 1)
        value_preds = self.value_pred.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        action_prob = self.action_prob.reshape(-1, self.action_prob.shape[-1])
        advantage = self.advantage.reshape(-1, 1)

        for j in range(num_batch):
            if (j + 1) * self.batch_size < self.episode_len:
                idx = indices[j * self.batch_size: (j+1) * self.batch_size]
                obs_batch = obs[idx]
                share_obs_batch = share_obs[idx]
                actions_batch = actions[idx]
                value_pred_batch = value_preds[idx]
                return_batch = returns[idx]
                action_prob_batch = action_prob[idx]
                advantage = advantage[idx]
                yield share_obs_batch, obs_batch, actions_batch, value_pred_batch, return_batch, action_prob_batch, advantage

    def __len__(self):
        return self.step