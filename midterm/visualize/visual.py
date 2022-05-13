import numpy as np
from sched import scheduler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.nn.parallel
import copy
import sys
sys.path.append('..')
from utils import *


transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
train_image = datasets.CIFAR100('../cifar100', train=True, download=True, transform=transform)
train_loader = DataLoader(train_image, batch_size=64, shuffle=False, num_workers=4)
ToPIL = transforms.ToPILImage()
for X, y in train_loader:
    for i in range(3):
        img = X[i]
        pic = ToPIL(img)
        pic.save('baseline' + str(i) + '.png')
    break

for X, y in train_loader:
    X = copy.deepcopy(X)
    lam = np.random.beta(1, 1)
    rand_index = torch.randperm(X.size()[0])
    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
    X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
    for i in range(3):
        img = X[i]
        pic = ToPIL(img)
        pic.save('cutmix' + str(i) + '.png') 
    break

for X, y in train_loader:
    X = copy.deepcopy(X)
    lam = np.random.beta(1, 1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
    X[:, :, bbx1:bbx2, bby1:bby2] = 0.0
    for i in range(3):
        img = X[i]
        pic = ToPIL(img)
        pic.save('cutout' + str(i) + '.png')
    break

for X, y in train_loader:
    X = copy.deepcopy(X)
    lam = np.random.beta(1, 1)
    rand_index = torch.randperm(X.size()[0])
    X = X * lam + X[rand_index] * (1. - lam)
    for i in range(3):
        img = X[i]
        pic = ToPIL(img)
        pic.save('mixup' + str(i) + '.png')
    break