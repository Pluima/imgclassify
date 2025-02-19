import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from  torch import  nn,optim
from  Resnet import ResNet

device = torch.device('cuda')
def evaluate(model,loader):
    correct =0
    total = len(loader.dataset)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred,y).sum().float().item()
    return correct/total

def main():
    batsz = 640
    cifar_train = datasets.ImageNet('imagenet',split='train',transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ]),download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batsz,shuffle=True)

    cifar_test = datasets.ImageNet('imagenet', 'val', transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batsz, shuffle=True)
    x,label = iter(cifar_train).__next__()
    print('x:',x.shape,'Label:',label.shape)
    best_acc=0
    model = ResNet(10).to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    opt= optim.Adam(model.parameters(),lr=1e-3)
    for echo in range(1000):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            x,label=x.to(device),label.to(device)
            logits= model(x)
            loss = criteon(logits,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(echo,':',loss.item())
        model.eval()
        if echo%2==0:
            with torch.no_grad():
                total_cor = 0
                total_num = 0
                for x, label in cifar_test:
                    x, label = x.to(device), label.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    total_cor += torch.eq(pred, label).float().sum().item()
                    total_num += x.size(0)
                acc = total_cor / total_num
                if(acc>best_acc):
                    best_acc=acc
                    torch.save(model.state_dict(), 'best_cifar.mdl')
                    print(echo, "best_acc:", best_acc)






if __name__=='__main__':
    main()