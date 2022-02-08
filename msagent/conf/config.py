
import torch
import numpy as np
import argparse
import os
import setproctitle
from pprint import pprint

def t2n(x):
    return x.detach().cpu().numpy()

# try:
#     import gfootball
# except ModuleNotFoundError:
#     raise RuntimeError("the gfootball module shoule be installed before")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_config():
    parser = argparse.ArgumentParser(description='train_group')
    parser.add_argument("--algorithm", type=str, default='mappo', choices=['mappo'])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--share_policy", action='store_false', default=True, help='whether the agent share the same policy')
    parser.add_argument("--env_id", default='11_vs_11_easy_stochastic',type=str,help='the env scenario')
    parser.add_argument("--num_left_agents", default=2, type=int)
    parser.add_argument("--num_right_agents", default=2, type=int)
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
    parser.add_argument("--log_dir", type=str, default='log')
    parser.add_argument("--use_all_reduce", type=bool, default=False)

    parser.add_argument("--env_contains_ma", type=int, default=2)
    parser.add_argument("--env_contains_me", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--main_agent", type=int, default=2)
    parser.add_argument("--main_exploiter", type=int, default=2)
    return parser


