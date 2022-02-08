import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from itertools import chain
import torch


class MLP(nn.Module):

    def __init__(self, input_shape, hidden_size, output_shape):
        super(MLP, self).__init__()
        if np.isscalar(hidden_size):
            hidden_size = [hidden_size]
        
        shapes = [input_shape] + hidden_size + [output_shape]

        in_dims = shapes[:-1]
        out_dims = shapes[1:]
        
        self.layer_names = ["mlp_{}".format(i+1) for i in range(len(hidden_size) + 1)]
        self.layer_func_names = ['relu_{}'.format(i+1) for i in range(len(hidden_size) + 1)]
        self.names = list(chain.from_iterable(zip(self.layer_names,self.layer_func_names)))
        self.archs = list()

        for in_dim, out_dim in zip(in_dims, out_dims):
            self.archs.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            self.archs.append(nn.ReLU())
        
        self.model = nn.Sequential(OrderedDict(list(zip(self.names, self.archs))))

        self.key = "mlp_block"

    def forward(self, x):
        out = self.model(x)
        return out
    


if __name__ == '__main__':
    o = MLP(10,[20,30],6)
    for module in o.modules():
        print(module)
    data = torch.from_numpy(np.arange(30).reshape(3,10)).float()
    ans = o(data)
    print(ans.shape)
