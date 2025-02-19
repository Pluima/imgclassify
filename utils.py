import  torch
from  torch import nn
class Faltten(nn.Module):
    def __init__(self):
        super(Faltten,self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)