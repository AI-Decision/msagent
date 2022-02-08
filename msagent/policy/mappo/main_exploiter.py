import numpy as np
import zmq
import torch
import pickle
from .algo import MAPPOAlgo
from msagent.leagues.coordinator import pfsp
from msagent.leagues.coordinator import PayOff
from time import sleep
import os

class MainAgent(object):
    def __init__(self, check_threshold, algo):
        super(MainAgent, self).__init__()
        self.checkpoint_step = 0
        self.gen = 1
        self.check_threshold = check_threshold
        self.policy_key = "main_agent_{}".format(self.gen)
        self.context = zmq.Context()
        self.algo = algo

    def pfsp_branch(self):
        historicals = [player for player in self.payoff.player]
        win_rates = self.payoff[self, historicals]
        return np.random.choice(historicals, p=pfsp(win_rates, weighting="squared"))

    def self_branch(self, opponent):
        if self.payoff[self, opponent] > 0.3:
            return opponent
        
        historicals = [player for player in self.payoff.players]
        win_rates = self.payoff[self, historicals]
        return np.random.choice(historicals, p=pfsp(win_rates, weighting="variance"))
    
    def create_checkpoint(self):
        self.pool_req_sock = self.context(zmq.REQ)
        self.pool_req_sock.connect("tcp://127.0.0.1:3421")
        self.pool_req_sock.send_string("write", zmq.SNDMORE)
        self.pool_req_sock.send_pyobj(self.agent)
        msg = self.pool_req_sock.recv_string()
        self.gen += 1

    def get_match(self):
        coin_toss = np.random.random()

        if coin_toss < 0.5:
            return self.pfsp_branch()
        
        main_agents =  [player for player in self.payoff.players if isinstance(player, MainAgent)]

        opponent = np.random.choice(main_agents)

        if coin_toss < 0.5 + 0.15:
            request = self.verfify_branch(opponent)
            return request
        
        return self.self_branch(opponent)

    def learn(self, world_size, rank,  data, backend='nccl', init_method='tcp://127.0.0.1:6900'):
        torch.distributed.init_process_group(backend=backend,
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size,
                                             group='ma_learner')
        torch.cuda.set_device(rank)
        device = torch.device("cuda:{}".format(rank))
        ac_loss, cr_loss, _, _ = self.algo.update(data)
        torch.distributed.destroy_process_group()
        return ac_loss, cr_loss, _, _

    def run(self, rank):
        while True:
            self.create_checkpoint()
            data = self.data_sock.recv_pyobj()
            world_size = os.environ['world_size']
            self.learn(world_size, rank, data)
            self.publish()


    def publish(self, broad_prot):
        params = self.algo.policy.state_dict()
        context = zmq.Context()
        pub_sock = context.socket(zmq.PUB)
        pub_sock.bind("tcp://*:{}".format(broad_prot))
        topic = 'MA_update'
        pub_sock.send_multipart([topic.encode('ascii'), pickle.dumps(params)])
        print("ready to send data again")

