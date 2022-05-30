import numpy as np
import os
from cv2 import cv2
import math
import random
import copy
from numba import jit
import sys
import mynet
import torch
import time

globe_Qy = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])
globe_Qc = np.array([
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

def psnr(img1,img2):
    mse = torch.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return (10 * torch.log10(PIXEL_MAX / mse)).item()

def normalJpegCompress(image,q):
    # input
    image_y,image_z,image_x = image.shape
    mypic = image.cuda()
    
    # rgb2yuv444
    x = mypic.permute(2,0,1).reshape(3,-1)
    x = x.cuda()
    w = globe_w.cuda()
    b = globe_b.cuda()
    YUV = torch.matmul(w,x) + b

    # dct and quatanization
    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i==0:
                a = torch.tensor(1/8).sqrt()
            else:
                a = torch.tensor(1/4).sqrt()
            A[i,j] = a * math.cos(math.pi*(j+0.5)*i/8)
    YUV = YUV.reshape(image_x,image_y,image_z)
    A = torch.Tensor(A)
    A = A.cuda()

    q_dct = torch.FloatTensor(image_x,image_y,image_z).fill_(0)
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    q_dct = q_dct.cuda()
    Qy = torch.Tensor(globe_Qy*s).cuda()
    Qc = torch.Tensor(globe_Qc*s).cuda()

    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y*image_z//64,1)
        else:
            Q_mat = Qc.repeat(image_y*image_z//64,1)
        a = torch.cat(YUV[i].split(8,0),1)
        a = torch.matmul(A,a)
        a = torch.cat(a.split(8,1),0)
        a = torch.div(torch.matmul(a,A.t()),Q_mat)
        a = torch.cat(a.split(8,0),1)
        q_dct[i] = torch.cat(a.split(image_z,1),0)

    # round
    quantanized_dct = q_dct.round()

    # i-DCT
    i_YUV = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        a = quantanized_dct[i] * Q_mat
        a = torch.cat(a.split(8,0),1)
        a = torch.matmul(A.t(),a)
        a = torch.cat(a.split(8,1),0)
        a = torch.matmul(a,A)
        a = torch.cat(a.split(8,0),1)
        i_YUV[i] = torch.cat(a.split(image_z,1),0)

    # YUV2RGB
    i_YUV = i_YUV.reshape(3,-1)
    i_x = torch.matmul(w.inverse(),i_YUV - b)
    real_input = torch.clamp(i_x,0,255).reshape(image_x,image_y,image_z).permute(1,2,0)
    return real_input.int()

def RGBToQdct(image,q):
    # input
    image_y,image_z,image_x = image.shape
    mypic = image.cuda()
    
    # rgb2yuv444
    x = mypic.permute(2,0,1).reshape(3,-1)
    x = x.cuda()
    w = globe_w.cuda()
    b = globe_b.cuda()
    YUV = torch.matmul(w,x) + b

    # dct and quatanization
    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i==0:
                a = torch.tensor(1/8).sqrt()
            else:
                a = torch.tensor(1/4).sqrt()
            A[i,j] = a * math.cos(math.pi*(j+0.5)*i/8)
    YUV = YUV.reshape(image_x,image_y,image_z)
    A = torch.Tensor(A)
    A = A.cuda()

    q_dct = torch.FloatTensor(image_x,image_y,image_z).fill_(0)
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    q_dct = q_dct.cuda()
    Qy = torch.Tensor(globe_Qy*s).cuda()
    Qc = torch.Tensor(globe_Qc*s).cuda()

    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y*image_z//64,1)
        else:
            Q_mat = Qc.repeat(image_y*image_z//64,1)
        a = torch.cat(YUV[i].split(8,0),1)
        a = torch.matmul(A,a)
        a = torch.cat(a.split(8,1),0)
        a = torch.div(torch.matmul(a,A.t()),Q_mat)
        a = torch.cat(a.split(8,0),1)
        q_dct[i] = torch.cat(a.split(image_z,1),0)
    return q_dct

def qdctToRGB(q_dct,q=50):
    # input
    image_x,image_y,image_z = q_dct.shape
    quantanized_dct = q_dct.cuda()

    # dct
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

    w = globe_w.cuda()
    b = globe_b.cuda()

    # quantinization
    if q<50:
        s=50/q
    else:
        s=2-2*q/100
    Qy = torch.Tensor(globe_Qy*s).cuda()
    Qc = torch.Tensor(globe_Qc*s).cuda()

    # i-DCT
    i_YUV = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        a = quantanized_dct[i] * Q_mat
        a = torch.cat(a.split(8,0),1)
        a = torch.matmul(A.t(),a)
        a = torch.cat(a.split(8,1),0)
        a = torch.matmul(a,A)
        a = torch.cat(a.split(8,0),1)
        i_YUV[i] = torch.cat(a.split(image_z,1),0)

    # YUV2RGB
    i_YUV = i_YUV.reshape(3,-1)
    i_x = torch.matmul(w.inverse(),i_YUV - b)
    real_input = torch.clamp(i_x,0,255).reshape(image_x,image_y,image_z).permute(1,2,0)
    return real_input.int()

def fastAdversarialRounding(advpic,mid_model,label,q,etaList):
    image_y,image_z,image_x = advpic.shape
    advQdct = RGBToQdct(advpic,q).detach().clone()
    dctGrad = mynet.getDctGrad(mid_model,advQdct,label,q)

    newDct = torch.where((advQdct.round()-advQdct)*dctGrad>0,advQdct.round(),advQdct)
    roundFlag = torch.Tensor(image_x,image_y,image_z).fill_(-1).cuda()
    roundFlag = torch.where((advQdct.round()-advQdct)*dctGrad>0,roundFlag*-1,roundFlag)
    grad_q_pic = mynet.get_x_to_dct_grad(mid_model,new_pic,label,q)

    d1 = newDct.ceil() - newDct
    d2 = newDct - newDct.floor()
    d = torch.where((d1-d2)==0,torch.Tensor(image_x,image_y,image_z).fill_(1e-10).cuda(),(d1-d2).abs())

    Qy = (torch.Tensor(globe_Qy)).cuda()
    Qc = (torch.Tensor(globe_Qc)).cuda()
    score = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        score[i] = dctGrad[i].abs()/(d[i]*d[i]*Q_mat*Q_mat)
    
    posSet = (roundFlag<0).nonzero()
    score_list = score[posSet[:,0],posSet[:,1],posSet[:,2]]

    newpic_list = []
    dctTowardGrad = torch.where(dctGrad>=0,newDct.ceil(),newDct.floor())
    for etaFactor in etaList:
        tmp_advQdct = newDct.clone()
        totalNum = math.floor(len(score_list) * etaFactor)
        _,indices = torch.topk(score_list,k=totalNum,largest=True)
        pos_group = torch.index_select(posSet,0,indices)
        tmp_advQdct[pos_group[:,0],pos_group[:,1],pos_group[:,2]] = dctTowardGrad[pos_group[:,0],pos_group[:,1],pos_group[:,2]]
        tmp_advQdct = tmp_advQdct.round()
        new_my_pic = qdctToRGB(tmp_advQdct,q).float()
        newpic_list.append(new_my_pic)
    return newpic_list

def iterativeAdversarialRounding(advpic,mid_model,original_label,target_label,q,etaList):
    image_y,image_z,image_x = advpic.shape
    advQdct = RGBToQdct(advpic,q).detach().clone()
    dest = advQdct.round()
    adv1 = advQdct.ceil()
    adv2 = advQdct.floor()
    dctGrad,loss1,loss2 = mynet.getDctGradWithOurLoss(mid_model,advQdct,dest,target_label,q)
    dctGrad *= -1.0
    max_iters = mynet.opt.max_iters
    learning_rate = mynet.opt.learning_rate
    times = 0
    miu = mynet.opt.miu
    momentumGrad = torch.Tensor(image_x,image_y,image_z).fill_(0).cuda()
    while times < max_iters:
        times += 1
        momentumGrad = miu * momentumGrad + dctGrad/dctGrad.abs().sum()
        t0_adv = advQdct + learning_rate * momentumGrad.sign()
        t1_adv = torch.where((advQdct==adv1)|(advQdct==adv2),advQdct,t0_adv)
        t2_adv = torch.where(t1_adv>adv1,adv1,t1_adv)
        advQdct = torch.where(t2_adv<adv2,adv2,t2_adv)
        dctGrad,loss1,loss2 = mynet.getDctGradWithOurLoss(mid_model,advQdct,dest,target_label,q)
        dctGrad *= -1.0
    
    adv1 = advQdct.ceil()
    adv2 = advQdct.floor()
    
    dctGrad = -1.0 * mynet.getDctGrad(mid_model,advQdct,target_label,q)

    advQdct = torch.where((advQdct.round()-advQdct)*dctGrad>0,advQdct.round(),advQdct)

    dctGrad = -1.0 * mynet.getDctGrad(mid_model,advQdct,target_label,q)
    d1 = adv1 - advQdct
    d2 = advQdct - adv2
    d = torch.where((d1-d2)==0,torch.Tensor(image_x,image_y,image_z).fill_(1e-10).cuda(),(d1-d2).abs())
    Qy = (torch.Tensor(globe_Qy)).cuda()
    Qc = (torch.Tensor(globe_Qc)).cuda()
    score = torch.FloatTensor(image_x,image_y,image_z).fill_(0).cuda()
    for i in range(3):
        if i == 0:
            Q_mat = Qy.repeat(image_y//8,image_z//8)
        else:
            Q_mat = Qc.repeat(image_y//8,image_z//8)
        score[i] = dctGrad[i].abs()/(d[i]*d[i]*Q_mat*Q_mat)

    roundFlag = torch.Tensor(image_x,image_y,image_z).fill_(-1).cuda()
    roundFlag = torch.where((advQdct==adv1)|(advQdct==adv2),roundFlag*-1,roundFlag)
    posSet = (roundFlag<0).nonzero()
    score_list = score[posSet[:,0],posSet[:,1],posSet[:,2]]

    newpic_list = []
    dctTowardGrad = torch.where(dctGrad>=0,advQdct.ceil(),advQdct.floor())
    for etaFactor in etaList:
        tmp_advQdct = advQdct.clone()
        totalNum = math.floor(len(score_list) * etaFactor)
        _,indices = torch.topk(score_list,k=totalNum,largest=True)
        pos_group = torch.index_select(posSet,0,indices)
        tmp_advQdct[pos_group[:,0],pos_group[:,1],pos_group[:,2]] = dctTowardGrad[pos_group[:,0],pos_group[:,1],pos_group[:,2]]
        tmp_advQdct = tmp_advQdct.round()
        new_my_pic = qdctToRGB(tmp_advQdct,q).float()
        newpic_list.append(new_my_pic)
    return newpic_list

if __name__ == '__main__':
    pass