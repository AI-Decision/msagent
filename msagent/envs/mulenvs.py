import cv2
import multiprocessing as mp

cv2.ocl.setUseOpenCL(False)

class MultipleEnvironment:
    def __init__(self, env_func, env_id, env_num):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(env_num)])
        self.envs = [env_func(env_id) for _ in range(env_num)]
        for index in range(env_num):
            process = mp.Process(target=self.run, args=(index, ))
            process.start()
            self.env_conns[index].close()
        
    
    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == 'step':
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == 'reset':
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError('current type can not be parsed')