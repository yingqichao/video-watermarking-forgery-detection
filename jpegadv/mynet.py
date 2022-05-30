import os
import numpy as np
import math
import torch
import jpgcompress
from config import DefaultConfig
import torchvision
import time

opt = DefaultConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_index

mean = (torch.FloatTensor([125.3, 123.0, 113.9]) / 255.0).cuda()
std = (torch.FloatTensor([63.0, 62.1, 66.7]) / 255.0).cuda()
subtrans = torchvision.transforms.Normalize(mean,std)

def my_trans(pic):
    return subtrans(pic/255.)

globe_Qy=torch.Tensor([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
globe_Qc=torch.Tensor([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])
globe_w = torch.Tensor([
    [0.299,0.587,0.114],
    [-0.168736,-0.331264,0.5],
    [0.5,-0.418688,-0.081312]
    ])
globe_b = torch.Tensor([
    [0],
    [128],
    [128]
    ])

def my_cal_dist(a,b,s=1):
    return torch.mean((a-b)**2)/s

def getDctGrad(my_model,mydct,label,q):
    # input
    image_x,image_y,image_z = mydct.shape
    my_input = mydct.detach().clone()
    my_input_cuda = my_input.cuda()
    my_input_cuda.requires_grad = True
    
    i_YUV = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    Qy = (globe_Qy*s).cuda()
    Qc = (globe_Qc*s).cuda()

    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i==0:
                a = torch.tensor(1/8).sqrt()
            else:
                a = torch.tensor(1/4).sqrt()
            A[i,j] = a * math.cos(math.pi*(j+0.5)*i/8)
    A = torch.Tensor(A)
    A = A.cuda()

    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        a = my_input_cuda[i] * Q_mat
        a = torch.cat(a.split(8,0),1)
        a = torch.matmul(A.t(),a)
        a = torch.cat(a.split(8,1),0)
        a = torch.matmul(a,A)
        a = torch.cat(a.split(8,0),1)
        i_YUV[i] = torch.cat(a.split(image_z,1),0)

    i_YUV_flatten = i_YUV.clone().reshape(image_x,-1)

    w = globe_w.cuda()
    b = globe_b.cuda()

    x = torch.matmul(w.inverse(),i_YUV_flatten - b)
    real_input = torch.clamp(x,0,255).reshape(image_x,image_y,image_z)

    loss_func = torch.nn.CrossEntropyLoss()
    
    t1 = my_model(my_trans(real_input).cuda().unsqueeze(0))
    my_output = t1.reshape(1,-1)

    target = torch.tensor([int(label)]).cuda()
    loss = loss_func(my_output,target)
    x.retain_grad()
    loss.backward()
    return my_input_cuda.grad

def getDctGradWithOurLoss(my_model,mydct,destdct,label,q):
    # input
    image_x,image_y,image_z = mydct.shape
    my_input = mydct.detach().clone()
    my_input_cuda = my_input.cuda()
    my_input_cuda.requires_grad = True
    destdct_cuda = destdct.cuda()
    
    i_YUV = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    Qy = (globe_Qy*s).cuda()
    Qc = (globe_Qc*s).cuda()

    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i==0:
                a = torch.tensor(1/8).sqrt()
            else:
                a = torch.tensor(1/4).sqrt()
            A[i,j] = a * math.cos(math.pi*(j+0.5)*i/8)
    A = torch.Tensor(A)
    A = A.cuda()

    myqdct = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    destqdct = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        myqdct[i] = my_input_cuda[i] * Q_mat
        destqdct[i] = destdct_cuda[i] * Q_mat
        a = torch.cat((my_input_cuda[i] * Q_mat).split(8,0),1)
        a = torch.matmul(A.t(),a)
        a = torch.cat(a.split(8,1),0)
        a = torch.matmul(a,A)
        a = torch.cat(a.split(8,0),1)
        i_YUV[i] = torch.cat(a.split(image_z,1),0)
    
    i_YUV_flatten = i_YUV.clone().reshape(3,-1)
    
    w = globe_w.cuda()
    b = globe_b.cuda()

    x = torch.matmul(w.inverse(),i_YUV_flatten - b)
    real_input = torch.clamp(x,0,255).reshape(image_x,image_y,image_z)
    
    loss_func = torch.nn.CrossEntropyLoss()

    t1 = my_model(my_trans(real_input).cuda().unsqueeze(0))
    my_output = t1.reshape(1,-1)
    target = torch.tensor([int(label)]).cuda()
    dist = my_cal_dist(myqdct,destqdct,s)
    loss1 = loss_func(my_output,target)
    loss2 = opt.dist_para * dist
    loss = loss1 + loss2
    loss.backward()
    return my_input_cuda.grad,loss1.item(),dist.item()

if __name__ == '__main__':
    pass