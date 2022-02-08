import consul
import numpy as np
import os
import subprocess
import signal
import socket
import uuid
import time
from collections import defaultdict
import zmq
import threading
import random
import importlib

class ConsulService:

    _instance_lock = threading.Lock()

    def __init__(self, host='172.17.0.2', port=8500, token=None):
        self.host = host
        self.port = port
        self.token = token
        self.consul = consul.Consul(host=host, port=port, token=token)
        self.name2id = defaultdict(lambda: [])


    def register(self,name, service_id, address, port, tag=None):
        self.name2id[name].append(service_id)
        self.consul.agent.service.register(name,
                                           service_id,
                                           address,
                                           port,
                                           tag)

    def deregister(self, service_name):
        for service_id in self.name2id[service_name]:
            self.consul.agent.service.deregister(service_id)
            self.consul.agent.check.deregister(service_id)


    def get_service(self, service_name):
        service_id = self.name2id[service_name].pop(0)
        self.name2id[service_name].append(service_id)
        services = self.consul.agent.services()
        service = services.get(service_id)
        if not service:
            return None, None 
        addr = "{0}:{1}".format(service['Address'], service['Port'])
        return service, addr

    @classmethod
    def instance(cls, *args, **kwargs):
        with ConsulService._instance_lock:
            if not hasattr(ConsulService, "_instance"):
                ConsulService._instance = ConsulService(*args, **kwargs)
        return ConsulService._instance


regis_auth = ConsulService.instance()