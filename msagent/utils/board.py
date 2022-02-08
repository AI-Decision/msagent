import zmq
import os
import numpy as np
from collections import OrderedDict
from msagent.utils.logger import logger 
from copy import deepcopy
import time
from threading import Lock



class Board(object):

    _context = zmq.Context()

    def __init__(self, port, *kwargs):
        self.agt2name = OrderedDict([(0,"root")])
        self.name2agt = OrderedDict([("root", 0)])
        self.port = port
        self.winrate = np.zeros((1,1),dtype=np.float32)

    def prob_dist(self, win_rate):
        if not(isinstance(win_rate, np.ndarray) and win_rate.shape[0] == win_rate.shape[1]):
            raise TypeError("the winrate should be a square matrix")
        num = win_rate.shape[0]
        dp = np.zeros(1<< num, dtype=np.float32)
        dp[-1] = 1
        # no key word argument in range 
        for i in range((1<<num)-1, 0, -1):
            cnt = 0
            for j in range(0, num, 1):
                if i & (1<<j):
                    cnt += 1
            total = cnt * (cnt -1) / 2
            if not total:
                continue
            for p in range(0, num, 1):
                if((1 << p) & i) == 0:
                    continue
                for q in range(p+1, num, 1):
                    if ((1 << q) &i) == 0:
                        continue
                    dp[i^(1<<p)] += dp[i] * win_rate[q][p] / total
                    dp[i^(1<<q)] += dp[i] * win_rate[p][q] / total
        ans = [data for (idx, data) in enumerate(dp) if  idx > 0 and idx & (idx-1) == 0]
        return ans
    
    def preprocess(self, win_rate):
        norm_win_rate = deepcopy(win_rate)
        for i in range(0, norm_win_rate.shape[0], 1):
            for j in range(0, i, 1):
                total_sum = norm_win_rate[i, j] + norm_win_rate[j, i]
                norm_win_rate[i, j] = norm_win_rate[i, j] / total_sum
                norm_win_rate[j, i] = norm_win_rate[j, i] / total_sum
        return norm_win_rate
    
    def run(self):
        rep_sock = self._context.socket(zmq.REP)
        rep_sock.bind("tcp://*:{}".format(self.port))
        while True:
            
            msg = rep_sock.recv_string()

            if msg == "result":
                match_info = rep_sock.recv_string()
                winner, loser = match_info.split('-')
                assert isinstance(winner, str) and isinstance(loser, str),\
                    "the info of match should be string"
                if winner in self.name2agt and loser in self.name2agt:
                    self.winrate[self.name2agt[winner], self.name2agt[loser]] += 1
                else:
                    self.winrate = np.pad(self.winrate, (0, 1), 'constant', constant_values=0)
                    if winner not in self.name2agt:
                        self.name2agt[winner] = self.winrate.shape[0] - 1
                        self.agt2name[self.winrate.shape[0] - 1] = winner
                    if loser not in self.name2agt:
                        self.name2agt[loser] = self.winrate.shape[0] - 1
                        self.agt2name[self.winrate.shape[0] - 1] = loser
                    
                    self.winrate[self.name2agt[winner], self.name2agt[loser]] += 1
                logger.info()

            elif msg == "sample":
                role = rep_sock.recv_string()
                assert role.startswith("MA") or role.startswith("ME"), \
                        "only support role type in the range of [MA, ME] for now"
                norm_matrix = self.preprocess(self.winrate)
                oppo_dist = self.prob_dist(norm_matrix)
                if role.startswith("ME"):
                    agt_len = norm_matrix.shape[0]
                    cand = [idx for idx in range(agt_len) if not self.agt2name[idx].startswith("ME")]
                    mask = np.ones(agt_len)
                    mask[cand] = 0
                    oppo_dist = oppo_dist * mask
                    oppo_dist = oppo_dist / oppo_dist.sum()
                op_idx = np.random.choice(oppo_dist, size=1, replace=False, p=oppo_dist)[0]
                assert np.isscalar(op_idx), 'the op should be an integer'
                oppo = self.agt2name[op_idx]
                logger.info()
            else:
                raise NotImplementedError("the {} type message can not be supported now".format(msg))
            time.sleep(1)