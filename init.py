from multiprocessing import Process
from sys import setprofile, stderr
import numpy as np
import subprocess
import time
import setproctitle
import os
import signal
from msagent.services.consul_service import regis_auth
import importlib
from msagent.services.solver import Solver
from msagent.services.register_class import get_local_ip
import multiprocessing as mp
from msagent.conf.config import get_config
from msagent.policy.mappo.policy import MAPPOPolicy
import torch

def wait(second):
    assert np.isscalar(second), \
        'the second should be a digit'
    time.sleep(second)


class NamedPopen(subprocess.Popen):
    '''like subprocess.Popen, but return the obect with the attribute of member'''
    def __init__(self, name=None, *args, **kwargs):
        super(NamedPopen, self).__init__(*args, **kwargs)
        self.name = name


if __name__ == '__main__':
    
    parser = get_config()
    args = parser.parse_args()
    
    procs = []

    IP = get_local_ip()

    C_cmd = "consul agent -server -bind={0} -client=0.0.0.0 -bootstrap=1 \
            -data-dir=/tmp/ConsulData -node server".format(IP).split()

    con_p = NamedPopen(args=C_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  \
                            start_new_session=True, name='consul')

    procs.append(con_p.pid)

    wait(2)

    R_cmd = "redis-cli"

    red_p = NamedPopen(args=R_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                            start_new_session=True, name='redis')

    procs.append(red_p.pid)

    wait(2)

    main_pid = os.getpid()

    setproctitle.setproctitle("main process of msagent")

    procs.append(main_pid)

    sig_listen = [signal.SIGINT, signal.SIGTERM]

    def quit_func(sig, frame):
        try:
            for p in procs:
                os.killpg(os.getpgid(p), 9)

        except OSError as e:
            raise OSError("the subprocesses need to be terminated manually due to the {}".format(e))

    for sig in sig_listen:
        signal.signal(sig, quit_func)

    act_module = importlib.import_module('./policy/mappo/worker', '.')
    Worker = getattr(act_module, 'Worker')

    worker = Worker(MAPPOPolicy)

    aval_gpu_cnt = torch.cuda.device_count()

    assert args.num_ma <= aval_gpu_cnt and \
           args.num_me <= aval_gpu_cnt, "the number of learners are not consistent with current amount of gpus"


    lrn_module = importlib.import_module('./policy/mappo/main_agent', '.')
    magt = getattr(lrn_module, 'MainAgent')

    lrn_module = importlib.import_module('./policy/mappo/main_exploiter', '.')
    megt = getattr(lrn_module, 'MainExploiter')

    env_module = importlib.import_module('./policy/mappo/env_agent', '.')
    env_ma = getattr(env_module, 'EnvAgent')

    env_module = importlib.import_module('./policy/mappo/env_exploiter', '.')
    env_me = getattr(env_module, 'EnvExploiter')

    for i in range(args.workers):
        p = Process(target=worker.listen)
        p.daemon = True
        p.start()
        procs.append(p.pid)

    for i in range(args.main_agent):
        p = Process(target=magt.run, args=(i,))
        p.daemon = True
        p.start()
        procs.append(p.pid)

    for i in range(args.main_exploiter):
        p = Process(target=megt.run, args=(i,))
        p.daemon = True
        p.start()
        procs.append(p.pid)

    for i in range(args.env_contains_ma):
        p = Process(target=env_ma.run)
        p.daemon = True
        p.start()
        procs.append(p.pid)

    for i in range(args.env_contains_me):
        p = Process(target=env_me.run)
        p.daemon = True
        p.start()
        procs.append(p.pid)
    
    while True:
        pass