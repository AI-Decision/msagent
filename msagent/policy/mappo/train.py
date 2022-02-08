import torch
import numpy as np
import argparse
import os
import setproctitle
from env import create_env
from ..utils.buffer import MultiAgentReplayBuffer
from .algo import MAPPOAlgo
from pprint import pprint

def t2n(x):
    return x.detach().cpu().numpy()

class Recorder(object):
    def __init__(self,home='home',away='away'):
        self.home = home
        self.away = away

    def getresult(self):
        pass

    def record(self, msg):
        pass


def t2n(x):
    return x.detach().cpu().numpy()


class Recorder(object):
    def __init__(self,home='home',away='away'):
        self.home = home
        self.away = away

    def getresult(self):
        pass

    def record(self, msg):
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mappo')
    parser.add_argument("--algorithm", type=str, default='mappo', choices=['mappo'])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--share_policy", action='store_false', default=True, help='whether the agent share the same policy')
    parser.add_argument("--env_id", default='11_vs_11_easy_stochastic',type=str,help='the env scenario')
    parser.add_argument('--num_agents', type=int, default=2, help='number of players')
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_max_grad_clip", type=float, default=0.01)
    parser.add_argument("--use_adv_norm",type=bool, default=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--coef", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float,default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.2)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--log_dir", type=str, default='\\log')

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(" trigger the GPU mode")
        device = torch.device("cuda:0")
    
    else:
        device = torch.device("cpu")
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    setproctitle.setproctitle("the main process")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env = create_env(env_id=args.env_id, left_agent=args.num_agents)

    algo = MAPPOAlgo(args, device=device)
    buffer = MultiAgentReplayBuffer(args)

    for episode in range(args.episodes):
        obs_lst = env.reset()
        for step in range(args.episode_length):
            act_lst, prob_lst= algo.policy.get_action_prob(obs_lst)
            share_obs_lst = np.reshape(obs_lst, (1, -1)).repeat(args.num_agents, axis=0)
            val_lst = algo.policy.get_value(share_obs_lst)
            next_obs_lst, reward_lst, done_lst, info = env.step(t2n(act_lst))

            done_lst = [done_lst] * args.num_agents
            data = share_obs_lst, obs_lst, t2n(act_lst), t2n(prob_lst), t2n(val_lst), reward_lst, done_lst
            buffer.insert(*data)
        buffer.compute_return()
        train_info = algo.train(buffer)
        pprint(train_info)