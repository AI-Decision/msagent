import torch

import horovod.torch as hvd

__all__ = ['Memory']


class Memory(object):
    def __init__(self, momentum=0.9, nesterov=False,
                 gradient_clipping=None, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}
    
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing sgd memory")
        for name, param in named_parameters:
            self.momentums[name] = torch.zeros_like(param.data)
            self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone() 

    def update(self, name, ctx):
        """Update the momentums."""
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(0, indices, 0)
        self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]