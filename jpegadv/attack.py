import os
import numpy as np
import jpgcompress
import mynet
import copy
import shutil
import time
import imageio
import math
import torch
import torchvision
import cv2
from config import DefaultConfig
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch import nn
from torch import cuda
from torch import optim
from torch import save
import foolbox

mean = (torch.FloatTensor([125.3, 123.0, 113.9]) / 255.0).cuda()
std = (torch.FloatTensor([63.0, 62.1, 66.7]) / 255.0).cuda()
subtrans = torchvision.transforms.Normalize(mean,std)

class my_network_2(nn.Module):
    def __init__(self,model):
        super(my_network_2,self).__init__()
        self.model = model
    def forward(self,input):
        return self.model(my_trans(input[0].cuda()).unsqueeze(0)).reshape(1,-1)

def my_trans(pic):
    return subtrans(pic/255.)

def mypredict(model,image):
    return model(my_trans(image.cuda().permute(2,0,1)).unsqueeze(0))

def gradient(model,image,label):
    my_input = image.detach().clone().cuda()
    my_input.requires_grad = True
    output = mypredict(model,my_input).reshape(1,-1)
    loss_func = torch.nn.CrossEntropyLoss()
    target = torch.tensor([int(label)]).cuda()
    loss = loss_func(output,target)
    loss.backward()
    return my_input.grad

def my_IGSM(model,image,label):
    myepsilon = mynet.opt.myepsilon
    steps = mynet.opt.steps
    epsilon = myepsilon / steps
    perturbed = image.cuda().clone()
    for _ in range(steps):
        g = gradient(model,perturbed,label)
        delta = torch.sign(g) * 255 * epsilon
        perturbed = perturbed + 1.0 * delta
        perturbed = torch.clamp(perturbed,0,255)
    return perturbed

def my_IGSM_T(model,image,target):
    myepsilon = mynet.opt.myepsilon
    steps = mynet.opt.steps
    epsilon = myepsilon / steps
    perturbed = image.cuda().clone()
    for _ in range(steps):
        g = gradient(model,perturbed,target)
        g = torch.sign(g) * 255 * epsilon
        perturbed = perturbed - 1.0 * g
        perturbed = torch.clamp(perturbed,0,255)
    return perturbed

def PGD(model,image,target):
    perturbed = image.cuda().clone().permute(2,0,1)
    fmodel = foolbox.PyTorchModel(my_network_2(model).cuda().eval(),bounds=(0,255))
    attack = foolbox.attacks.LinfPGD(
        steps=10,
        rel_stepsize=0.1,
        random_start=False
        )
    criterion = foolbox.Misclassification(torch.LongTensor([target]).cuda())
    a,perturbed,success = attack(fmodel,perturbed.unsqueeze(0),criterion,epsilons=3)
    perturbed = perturbed[0]
    return perturbed.permute(1,2,0)