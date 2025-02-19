# -*- coding: UTF-8 -*-
import os
import glob
import csv,random
import torch
import os,glob
import random,csv

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from  PIL import Image




class pho_loader():
    def __init__(self,poi,resize):
        super(pho_loader,self).__init__()
        self.poi=poi
        self.labels=[]
        self.images=self.load_csv('images.csv')
        self.resize=resize

    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.poi,filename)):
            images =[]
            for name in sorted(os.listdir(os.path.join(self.poi))):
                images += glob.glob(os.path.join(self.poi,name, "*.png"))
                images += glob.glob(os.path.join(self.poi, name,'*.jpg'))
                images += glob.glob(os.path.join(self.poi, name,'*.jpeg'))



            with open(os.path.join(self.poi,filename),mode='w',newline='')as f:
                writer = csv.writer(f)
                for img in images:
                    writer.writerow([img])
        images = []
        with open(os.path.join(self.poi,filename))as f:
            reader = csv.reader(f)
            for row in reader:
                img = row[0]
                images.append(img)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img=self.images[idx]
        # print(img)
        tf=transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize),int(self.resize))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        img = tf(img)

        return img



def main():

    import visdom
    import time
    viz=visdom.Visdom()
    db = pho_loader('unsorted_pics',224)
    x = next(iter(db))
    print(x.shape)
    viz.image(x,win='sample_x',opts=dict(title='sample_x'))
    loader = DataLoader(db,batch_size=1024)

if __name__ == '__main__':
    main()