import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils.buffer import SingleAgentReplayBuffer
from model import Network
import numpy as np

class DQNAlgo:

    def __init__(self, env, 
                memory_size,
                batch_size, 
                target_update, 
                epsilon_decay, 
                max_epsilon = 1.0,
                min_epsilon = 0.1,
                gamma = 0.99):

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.memory = SingleAgentReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())
        
        self.transition = list()
    

    def select_action(self, state):
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.cpu().numpy()
        self.transition = [state, selected_action]
        return selected_action
        
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)
        return next_state, reward, done    


    def update(self):
        samples = self.memory.sample_batch()
        loss = self.compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    
    def train(self, num_frames):
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)

            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
            
            if len(self.memory) >= self.batch_size:
                loss = self.update()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(self.min_epsilon, self.epsilon- \
                                    (self.max_epsilon - self.min_epsilon) * \
                                     self.epsilon_decay)
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self.target_hard_update()
        
        # self.env.close()

    def compute_loss(self, sample):

        state = torch.FloatTensor(sample['obs']).to(self.device)
        next_state = torch.FloatTensor(sample['next_obs']).to(self.device)
        action = torch.LongTensor(sample['acts'].reshape(-1,1)).to(self.device)
        reward = torch.FloatTensor(sample['rews'].reshape(-1,1)).to(self.device)
        done = torch.FloatTensor(sample["done"].reshape(-1, 1)).to(self.device)

        cur_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask)
        loss = F.smooth_l1_loss(cur_q_value, target)
        return loss

