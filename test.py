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

# device = 'cuda'
device = 'cpu'

model=VIT_B16_224().to(device)

if __name__ == '__main__':
    input_size = 224
    model=torch.load('./models/vit_model69.pth')
    transform_valid = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor()])
    img = Image.open('./data/test/cat.1.jpg')
   #  img = Image.open('./data/test/dog.1.jpg')
    img_ = transform_valid(img).unsqueeze(0) #拓展维度
    print(img_.shape)
    # print(model.device)
    # print(img_.device)
    preds = model(img_.to('cuda:0')) # (1, 1000)
    # print(preds)
    kname, indices = torch.max(preds,1)
    print("----kind=",indices)