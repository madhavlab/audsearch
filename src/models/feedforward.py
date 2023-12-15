import torch.nn as nn
import torch

class ProjectionHead1(nn.Module):
    def __init__(self, d=128, h=1024, u=32):
        super().__init__()
        assert h%d == 0, 'h must be divisible by d'
        v = h//d
        self.d = d
        self.h = h
        self.u = u
        self.v = v
        # print(f"d:{d}, h:{h}, u:{u}, v:{self.v}")
        self.linear1 = nn.Conv1d(d * v, d * u, kernel_size=(1,), groups=d)
        self.elu = nn.ELU()
        self.linear2 = nn.Conv1d(d * u, d, kernel_size=(1,), groups=d)
    
    def forward(self, x, norm=False):
        x = x.view(-1, self.h, 1)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        if norm:
            x = torch.nn.functional.normalize(x, p=2.0)
        return torch.squeeze(x)