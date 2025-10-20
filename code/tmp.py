import torch
from networks.hn import HN

x = torch.rand(size=[2, 1, 64, 64, 64]).cuda()
net = HN().cuda()

y =net(x)
