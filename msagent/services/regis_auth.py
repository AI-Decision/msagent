import json
import requests
import zmq
import consul
from random import randint


class RegisterClient():
    def __init__(self, host=None, port=None, token=None):
        self.host = host
        self.port = port
        self.token = token
        self.consul = consul.Consul(host=host, port=port)

    def register(self, name, service_id, address, port, tags=None, interval=None, httpcheck=None):
        self.consul.agent.service.register(name, service_id=service_id, address=address, port=port, tags=tags,
                                           interval=interval, httpcheck=httpcheck)

    def deregister(self, service_id):
        self.consul.agent.service.deregister(service_id)
        self.consul.agent.check.deregister(service_id)

    def getService(self, name):
        services = self.consul.agent.services()
        return services[0].get(name)

class LocalRegisterClient():

    def __init__(self, host='localhost', port='10600'):
        context = zmq.Context()
        self.rc_socket = context.socket(zmq.REQ)
        self.rc_socket.connect('tcp://{}:{}'.format(host, port))
    
    def register(self, name, service_id, address, port, tags=None):
        data = {'type': 'register', 'name': name, 'address': 'localhost', 'port': port, 'tags': tags}
        self.rc_socket.send_pyobj(data)
        ok = self.rc_socket.recv_pyobj()
        if not ok:
            raise RuntimeError
    
    def getService(self, name):
        data = {'type':'get', 'name':name,}
        self.rc_socket.send_pyobj(data)
        serivce_addr = self.rc_socket.recv_pyobj()
        return serivce_addr
    
    def get_port(self):
        data = {'type':'port'}
        self.rc_socket.send_pyobj(data)
        port = self.rc_socket.recv_pyobj()
        return port
        
        
        
if __name__ == '__main__':
    c = LocalRegisterClient('localhost', '10600')
    service_id = 'Messer' + '127.0.0.1' + ':' + str(10107)
    c.register('test1', 'test1', '127.0.0.1', 10107, ['provider'])
    print(c.getService('test1'))
