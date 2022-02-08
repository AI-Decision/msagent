import zmq
from msagent.services.consul_service import regis_auth
import pickle

class Solver:

    context = zmq.Context()
    
    def __init__(self, *args, **kwargs):
        pass

    def retrive(self, worker, fn_name, *args, **kwargs):
        _, use_ip = regis_auth.get_service(fn_name)
        sock = self.context.socket(zmq.DEALER)
        
        sock.connect("tcp://{0[0]}:{0[1]}".format(use_ip.split(":")))
        sock.send_multipart([pickle.dumps(worker), pickle.dumps(fn_name), pickle.dumps(args), pickle.dumps(kwargs)])

        if 'response' in kwargs and kwargs['response'] == True:
            ans = sock.recv_multipart()
            ans = pickle.loads(ans[0])
            return ans

