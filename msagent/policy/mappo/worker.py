import torch
import numpy as np
import zmq
import pickle
import threading
import time
import inspect
from msagent.services.register_class import register

class Worker:
    def __init__(self, policy, update_port=7772):
        self.policy = policy
        self.update_port = update_port
        self.lock = threading.Lock()
        bg_t = threading.Thread(target=self.listen, args=(self.policy.actor, self.lock))
        bg_t.setDaemon(True)
        bg_t.start() 
        time.sleep(2)


    def listen(self):
        context = zmq.Context()
        sub_sockt= context.socket(zmq.SUB)
        sub_sockt.connect("tcp://localhost:{}".format(self.update_port))
        filter = "MA_update"
        sub_sockt.setsockopt_string(zmq.SUBSCRIBE, filter)
        while True:
            topic, model_updated = sub_sockt.recv_multipart()
            assert topic == filter.encdoe("utf-8"), \
                "get the wrong topic during the subscribe process"
            model_updated = pickle.loads(model_updated)
            self.lock.acquire()
            self.policy.load_state_dict(model_updated.state_dict())
            print("update the model parameter")
            self.lock.release()
            time.sleep(2)

    @register
    def get_act(self, obs):
        self.lock.acquire()
        action, logit = self.policy.eval_actor_prob(obs)
        self.lock.release()
        return (action, logit)

    @register
    def get_val(self, obs):
        self.lock.acquire()
        value = self.policy.eval_critic_value(obs)
        self.lock.release()
        return value