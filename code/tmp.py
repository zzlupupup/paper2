import torch
import numpy as np
from torch import nn
from utils.ramps import sigmoid_rampup
import torch.nn.functional as F
from utils import unhn_util
from networks.hn import HN

pred = torch.rand(size=[2, 1, 64, 64, 64]).cuda()


net = HN(fusion_type='UNHN').cuda()
net(pred)



