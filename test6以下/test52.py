import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

# print(net[1].state_dict())

#访问参数
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# print(net.state_dict()['2.weight'].data)


#参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

