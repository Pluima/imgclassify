import numpy
import torch
from torch import nn
from  torch.utils.data import DataLoader,Dataset
# from Photo_sorter import pho_loader
# from Resnet import ResNet
# import os
import torchvision
from torchvision.models import  resnet34
import utils
from utils import Faltten
#
#
# device = torch.device('cuda')
# batsz = 256
# db=pho_loader('unsorted_pics',224)
# album_loader= DataLoader(db,batch_size=batsz,num_workers=8)
#
# def main():
#     pre_trained = resnet18(pretrained=True)
#     model = nn.Sequential(*list(pre_trained.children())[:-1],
#                           Faltten(),
#                           nn.Linear(512, 17)
#                           ).to(device)
#     model.load_state_dict(torch.load('best.mdl'))
#     for x in album_loader:
#         x=x.to(device)
#         with torch.no_grad():
#             logits = model(x)
#             pred = logits.argmax(dim=1)
#

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import os
import shutil

# 加载PyTorch模型
device = torch.device('cuda')
pre_trained = resnet34(pretrained=True)
model = nn.Sequential(*list(pre_trained.children())[:-1],
                          Faltten(),
                          nn.Linear(512, 17)
                          ).to(device)
model.load_state_dict(torch.load('best.mdl'))
model.eval()
transform = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((224,224)),
            # transforms.Resize((int(self.resize),int(self.resize))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                                ])

# 创建GUI窗口
root = tk.Tk()
root.title("Image Classifier")


# 创建文件夹选择按钮
def select_directory():
    folder_path = filedialog.askdirectory()
    if folder_path:
        classify_images_in_folder(folder_path)


folder_button = tk.Button(root, text="选择文件夹及其子文件夹", command=select_directory)
folder_button.pack()

# 显示图像和分类结果的标签
image_label = tk.Label(root)
image_label.pack()
result_label = tk.Label(root)
result_label.pack()


# 定义图像分类函数
def classify_images_in_folder(root_folder):
    for root_dir, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root_dir, filename)
                image = Image.open(image_path)
                image = transform(image)

                # 使用模型进行分类
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                    predicted_class = predicted.item()

                # 移动图像到相应的文件夹
                class_folder = str(predicted_class)
                class_folder_path = os.path.join(root_folder, class_folder)
                os.makedirs(class_folder_path, exist_ok=True)
                new_image_path = os.path.join(class_folder_path, filename)
                shutil.move(image_path, new_image_path)

                # 在GUI中显示图像和分类结果
                img = Image.open(new_image_path)
                img.thumbnail((200, 200))
                img = ImageTk.PhotoImage(img)
                image_label.configure(image=img)
                image_label.image = img
                result_label.config(text=f"分类结果: {predicted_class}")


root.mainloop()


#
# if __name__ == '__main__':
#     main()