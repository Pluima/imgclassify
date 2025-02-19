import torch
from torch import nn
from torch.nn import functional as fuc


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),

        )
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,60),
            nn.ReLU(),
            nn.Linear(60,10)
            )


    def forward(self, x):
        batsz=x.size(0)
        x=self.conv_unit(x)
        x=x.view(batsz,16*5*5)
        logits=self.fc_unit(x)
        return logits
def main():
    net = Lenet5()


if __name__ == '__main__':
    main()