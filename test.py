from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from model_vit import VIT_B16_224
import os
from linformer import Linformer
from vit_pytorch.efficient import ViT

# device = 'cuda'
device = 'cpu'

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# model=VIT_B16_224().to(device) # the model could be the ViT model

if __name__ == '__main__':
    input_size = 224
    model=torch.load('./models/vit_model.pth')
   #  model=torch.load('./models/vit_model_linear.pth')
    transform_valid = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor()])
    img_list = os.listdir('./data/train')
    for filename in img_list:
       img = Image.open('./data/train/'+filename)
       img_ = transform_valid(img).unsqueeze(0) #拓展维度
       preds = model(img_.to('cuda:0')) # (1, 1000)
       kname, indices = torch.max(preds,1)
       print("-------img_name=",filename,"----kind=",indices)

   #  single image is tested as follows:
   #  img = Image.open('./data/test/cat.1.jpg')
   #  img_ = transform_valid(img).unsqueeze(0) #拓展维度
   #  preds = model(img_.to('cuda:0')) # (1, 1000)
   #  kname, indices = torch.max(preds,1)
   #  print("----kind=",indices)