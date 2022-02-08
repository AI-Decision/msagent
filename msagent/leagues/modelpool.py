import cloudpickle
import zmq
import pprint
import time
import multiprocessing as mp
import random
import torch
import os
import redis

class ModelPool(object):

    _context = zmq.Context()

    def __init__(self, rep_port, db_info={}, *args, **kwargs):
        super(ModelPool, self).__init__()
        self.ip_addr = db_info.get('ip', '172.17.0.2')
        self.port = db_info.get('port', 6379)
        self.rep_port = rep_port
        self.pool = redis.ConnectionPool(host=self.ip_addr, port=self.port)
        self.init_check()
        self.redis_cli = redis.Redis(connection_pool=self.pool)

    def init_check(self):
        try:
            redis.Redis(host=self.ip_addr, port=self.port, db=0).ping()
        except redis.ConnectionError:
            raise RuntimeError("should init the redis-server first")
    
    def response(self):
        rep_sock = self._context.socket(zmq.REP)
        rep_sock.bind("tcp://*:{}".format(self.rep_port))
        while True:
            msg = rep_sock.recv_string()
            if msg == 'read':
                key = rep_sock.recv_string()
                ser_model = self.redis_cli.get(key)
                if ser_model is None:
                    print("use init model......")
                    des_model = self.initmodel
                else:
                    des_model = cloudpickle.loads(ser_model)
                rep_sock.send_pyobj(des_model)
            elif msg == 'write':
                model = rep_sock.recv_pyobj()
                ser_model = cloudpickle.dumps(model)
                self.redis_cli.set(model.key, ser_model)
                rep_sock.send_string("ok")
            else:
                raise NotImplementedError("the {} type message can not be parsed".format(msg))

    def delete(self, key):
        self.redis_cli.delete(key)

    def close(self):
        self.redis_cli.flushdb()
        self.redis_cli.shutdown(nosave=True)
        print("redis shutdown")


if __name__ == '__main__':
    mpool = ModelPool(rep_port=7784)
    mp_proc = mp.Process(target=mpool.response, args=(), daemon=False)
    mp_proc.start()
