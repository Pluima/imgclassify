import torch
from torch import nn
from torch.nn import functional as func
class Resblk(nn.Module):
    def __init__(self,Ch_in,Ch_out,stride=1):

        super(Resblk,self).__init__()
        self.conv1 = nn.Conv2d(Ch_in,Ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(Ch_out)
        self.conv2 = nn.Conv2d(Ch_out,Ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(Ch_out)
        self.extra=nn.Sequential()
        if Ch_out != Ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(Ch_in,Ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(Ch_out)
            )
    def forward(self,x):
        out = func.relu(self.bn1(self.conv1(x)))
        out = func.relu(self.bn2(self.conv2(out)))
        out = self.extra(x) + out
        out = func.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,num_class):
        super(ResNet,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
        self.blk1 = Resblk(64,128 ,stride=2)
        self.blk2 = Resblk(128,256,stride=2)
        self.blk3 = Resblk(256,512,stride=2)
        self.blk4 = Resblk(512,512,stride=1)
        self.outlayer = nn.Linear(512*1*1,num_class)
    def forward(self,x):
        x = func.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x= func.adaptive_avg_pool2d(x,[1,1])

        x= x.view(x.size(0),-1)
        x=self.outlayer(x)
        return x
def main():
    blk = Resblk(64,128)
    tmp=torch.randn(2,64,224,224)
    out = blk(tmp)
    print('block:',out.shape)
    x = torch.randn(2,3,224,224)
    model = ResNet(15)
    out = model(x)
    print(out.shape)
    p = sum(map(lambda p:p.numel(),model.parameters()))
    print('paras',p)
if __name__=='__main__':
    main()