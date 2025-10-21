import torch
from torch import nn
import torch.nn.functional as F

x = torch.tensor([
    [[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ]],
    [[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ]]
])
print(x.shape)

avg_x = F.avg_pool2d(x, kernel_size=2, stride=2)
print(avg_x)
print(avg_x.shape)

B, C, P, _ = avg_x.shape
L = P ** 2
seq_x = avg_x.view(B, 1, L).expand(B, L, L).view(B, 1, L, L).expand(B, 2, L, L).contiguous().view(2 * B, L, L)
print(seq_x)
print(seq_x.shape)

