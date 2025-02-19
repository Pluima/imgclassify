import torch
from torch import optim,nn
import visdom
import  torchvision
from torch.utils.data import DataLoader,Dataset
from Photo import Photos
from Resnet import ResNet
from torchvision.models import resnet152
from  utils import Faltten
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
batsz = 256
lr = 5e-5
epochs = 100
device = torch.device('cuda')
torch.manual_seed(4444)

#train_db = Photos('Photos_train',224,mode='train')
#val_db = Photos('Photos_train',224,mode='val')
# test_db = Photos('Photos_train',224,mode='test')
#train_loader = DataLoader(train_db,shuffle=True,batch_size=batsz,num_workers=8)
#val_loader = DataLoader(val_db,batch_size=batsz,num_workers=4)
# test_loader = DataLoader(test_db,batch_size=batsz,num_workers=4)
viz = visdom.Visdom()
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
    model = ResNet(15).to(device)
    #pre_trained = resnet152(pretrained=True)
    # model = nn.Sequential(*list(pre_trained.children())[:-1],
    #                       Faltten(),
    #                       nn.Linear(512,14)
    #                       ).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optmizer = optim.Adam(model.parameters(),lr=lr,weight_decay=0.00125)
    criteon = nn.CrossEntropyLoss()
    best_acc =0
    best_epoch = 0
    global_state=0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss = criteon(logits,y)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
            viz.line([loss.item()], [global_state], win='loss',update='append')
            global_state+=1


        if epoch%2 == 0:
            val_acc=evaluate(model,val_loader)
            viz.line([val_acc], [global_state], win='val_acc', update='append')
            if(val_acc > best_acc):
                best_acc=val_acc
                best_epoch=epoch
                torch.save(model.state_dict(),'best.mdl')
                print('best acc:',best_acc,'best epoch',best_epoch)



    # model.load_state_dict(torch.load('best.mdl'))
    # print('model loaded')
    # test_acc = evaluate(model,val_loader)
    # print('test_acc: ',test_acc)



if __name__ == '__main__':
    main()