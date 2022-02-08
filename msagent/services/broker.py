from msagent.utils.logger import logger
import zmq
import time
from collections import OrderedDict



class Broker(object):
    def __init__(self, fport, bport, log_file=None, *args, **kwargs):
        self.fport = fport
        self.bport = bport
        self.workers = OrderedDict()
        self.clients = {}
        self. msg_cache = []

    def run(self):
        self.context = zmq.Context()
        
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.bind("tcp://*:{}".format(self.fport))
        self.frontend.setsockopt(zmq.RCVHWM, 100)

        self.backend = self.context.socket(zmq.DEALER)
        self.backend.bind("tcp://*:{}".format(self.bport))
        self.backend.setsockopt(zmq.RCVHWM, 100)

        self.poll = zmq.Poller()
        self.poll.register(self.frontend, zmq.POLLIN)
        self.poll.register(self.backend, zmq.POLLIN)

        while True:
            socks = dict(self.poll.poll())
            curtime = time.time()

            if self.backend in socks and socks[self.backend] == zmq.POLLIN:
                worker_addr, client_addr, response = self.backend.recv_multipart()

                self.workers[worker_addr] = time.time()
                if client_addr in self.clients:
                    self.frontend.send_multipart([client_addr, response])
                    self.clients.pop(client_addr)
                
                else:
                    print("the client does not exist")
        
            while len(self.msg_cache) > 0 and len(self.workers) > 0:
                worker_addr, t = self.workers.popitem()

                if t - curtime > 1:
                    continue
                msg = self.msg_cache.pop(0)

                self.backend.send_multipart([worker_addr, msg[0], msg[1]])

            if self.frontend in socks and socks[self.frontend] == zmq.POLLIN:
                client_addr, request = self.frontend.recv_multipart()
                self.clients[client_addr] = 1

                while len(self.workers) > 0:
                    worker_addr, t = self.workers.popitem()

                    if t - curtime > 1:
                        continue

                    self.backend.send_multipart([worker_addr, client_addr, request])
                    break
                
                else:
                    self.msg_cache.append([client_addr, request])


    
