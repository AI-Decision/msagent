import zmq
import uuid
import threading
from msagent.services.consul_service import regis_auth
import pickle
import socket


def get_open_port(host='172.17.0.2'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def get_local_ip():
    hostname = socket.getfqdn()
    ip_address = socket.gethostbyname_ex(hostname)[2][0]
    return ip_address


class RegisterClass:

    name2fn = {}
    context = zmq.Context()
    svid2sock = dict()

    def __init__(self, name='name', id='id', address='address', port='port', tag=None):

        self.name = name
        self.service_id = name + str(id)
        self.address = address
        self.port = port
        self.tag = tag
        self.cls_name = self.__class__.__name__ 
        fn_list = self.__enroll()
        for f in fn_list:
            self.name2fn[f] = getattr(self, f)
    
    def __enroll(self):

        return list(filter(lambda m: (not m.startswith("__") and \
                                      not m.startswith("_" + self.cls_name) and \
                                      callable(getattr(self, m))), dir(self)))

    
    @classmethod
    def register(cls, *args, **kwargs):
        def decorator(fn):
            cls.name2fn[fn.__name__] = fn
            svid = fn.__name__ + str(uuid.uuid4())[:10]
            local_ip = get_local_ip()
            open_port = get_open_port(local_ip)
            regis_auth.register(fn.__name__, svid ,address=local_ip, port=open_port)
            sock = cls.context.socket(zmq.DEALER)
            sock.bind("tcp://*:{}".format(open_port))
            cls.svid2sock[svid] = sock
            t =  threading.Thread(target=cls.worker_func, args=(sock,))
            t.start()
            return fn
        return decorator(args[0])
    

    @classmethod
    def worker_func(cls,sock):
        while True:
            obj, fn_name, args, kwargs = sock.recv_multipart()
            obj = pickle.loads(obj)
            fn_name =pickle.loads(fn_name)
            args = pickle.loads(args)
            kwargs = pickle.loads(kwargs)
            fn = cls.name2fn[fn_name]
            ans = fn(obj, args[0])
            sock.send_multipart([pickle.dumps(ans)])


    def destroy(self):
        for sock in self.svid2sock.values():
            sock.destory()
        self.context.term()


register = RegisterClass.register
