import torch
from torch import nn

# print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
# print(torch.cuda.device_count())


def try_gpu(i=0):  #@save
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

X = torch.ones(2, 3, device=try_gpu())
print(X)