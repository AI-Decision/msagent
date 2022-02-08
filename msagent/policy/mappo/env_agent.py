import os
import zmq
from .algo import MAPPOAlgo
import torch
from .env import create_env
from ..utils.buffer import MultiAgentReplayBuffer
import numpy as np
import random
from .worker import Worker
from msagent.services.solver import Solver

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t2n(x):
    return x.detach().cpu().numpy()

class EnvAgent(object):

    _context = zmq.Context()

    def __init__(self, args):
        self.args = args
        self.alg = MAPPOAlgo(args, device=torch.device("cuda:0"))
        self.env = create_env(args.env_id, args.num_left_agents, args.num_right_agents)
        self.num_agents = args.num_left_agents + args.num_right_agents
        self.bd_port = args.bd_port
        self.buffer = MultiAgentReplayBuffer(args)
        self.worker = Worker(policy=self.alg.policy)
        self.solver = Solver()
        self.run()

    def run(self):
        result = list()
        bd_sock = self._context.socket(zmq.REQ)
        db_sock = self._context.socket(zmq.REQ)
        data_sock = self._context.socket(zmq.REQ)
        bd_sock.connect("tcp://localhost:{}".format(self.bd_port))
        db_sock.connect("tcp://localhost:{}".format(self.db_port))
        data_sock.connect("tcp://localhost:{}".fomrat(self.data_port))
        for episode in range(self.args.episodes):
            obs_lst = self.env.reset()
            bd_sock.send_string("sample", zmq.SNDMORE)
            role_name = bd_sock.send_string("MA")

            status = bd_sock.recv_string()

            if status != 'ok':
                raise RuntimeError("Can not get the Right opponent's name")

            db_sock.send_string("read", zmq.SNDMORE)
            op_policy = db_sock.send_string(role_name)

            if status != 'ok':
                raise RuntimeError("Can not get the Right opponent's policy")

            for step in range(self.args.episode_length):
                act_lst, prob_lst = self.solver.retrive(self.worker, 'get_act', 
                                                        obs_lst[:self.args.num_left_agents], 
                                                        **{'response':True})

                op_act_lst, op_prob_lst = op_policy.get_action_prob(obs_lst[self.args.num_left_agents:])
                share_obs_lst = np.reshape(obs_lst[:self.args.num_left_agents], 
                                            (1, -1)).repeat(self.num_left_agents, axis=0)                

                val_lst = self.solver.retrive(self.worker, 'get_val',
                                              obs_lst[:self.args.num_left_agents],
                                              **{'response':True})

                next_obs_lst, reward_lst, done_lst, info = self.env.step(np.append(t2n(act_lst), t2n(op_act_lst)))

                if info['score_reward']:
                    result.append(info['score_reward'])

                done_lst = [done_lst] * self.num_left_agents

                data = share_obs_lst, obs_lst[:self.args.num_left_agents], t2n(act_lst), \
                        t2n(prob_lst), t2n(val_lst), \
                        reward_lst[:self.args.num_left_agents], done_lst
                
                self.buffer.insert(*data)
            ans = sum(result)
            if ans > 0:
                outcome = "ma-me"
            elif ans < 0:
                outcome = "me-ma"
            else:
                outcome = "ma-ma"
            bd_sock.send_string("result", zmq.SNDMORE)
            bd_sock.send_string(outcome)

            if status != 'ok':
                raise RuntimeError("can not register the outcome")

            data_generator = self.buffer.sample()
            for data in data_generator:
                data_sock.send_pyobj(data)
                _ = data_sock.recv_string()

    
            