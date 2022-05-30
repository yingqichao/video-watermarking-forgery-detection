
import cv2

import torch
import torch.utils.data as data
# import utils.util as util
import util

import torchvision.transforms.functional as F
import pandas
from PIL import Image
import os
import openpyxl
import pandas as pd
import numpy as np
from tqdm import tqdm

import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_img(imgs, root_path,videopath, GT_size):
    GT_path = "{}/{}/{}/".format(root_path, videopath, imgs)
    Img_list = os.listdir(GT_path)
    Img_list.sort(key = lambda x: int(x[1:-4]))
    Video_GT = torch.zeros(3, len(Img_list), GT_size, GT_size)
    for i in range(len(Img_list)):
        img_GT = util.read_img("{}/{}/{}/{}".format(root_path, videopath, imgs, Img_list[i]))
        img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        ###### directly resize instead of crop
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),  interpolation=cv2.INTER_LINEAR)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()

        img_GT = torch.where(img_GT > 0, torch.ones_like(img_GT), torch.zeros_like(img_GT))
        Video_GT[:,i,:,:] = img_GT.unsqueeze(0)

    return Video_GT


def read_mask(imgs, root_path, videopath, GT_size):
    rate = []
    GT_path = "{}/{}/{}/".format(root_path, videopath, imgs)
    Img_list = os.listdir(GT_path)
    Img_list.sort(key=lambda x: int(x[1:-4]))
    Video_GT = torch.zeros(1 ,len(Img_list), GT_size, GT_size)
    for i in range(len(Img_list)):
        img_GT = util.read_img("{}/{}/{}/{}".format(root_path, videopath, imgs, Img_list[i]))
        img_GT = util.channel_convert(img_GT.shape[2], 'gray', [img_GT])[0]
        ###### directly resize instead of crop
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
        Video_GT[:,i,  :, :] = img_GT.unsqueeze(0)
        rate.append(torch.mean(img_GT))

    return Video_GT, sum(rate)/len(rate)

class DVDataset(data.Dataset):

    def __init__(self, root_path='/home/groupshare/DAVIS/', image_size=256, is_train=True):
        super(DVDataset, self).__init__()
        self.is_train = is_train
        self.root_path = root_path
        self.index = 0
        self.label_dict = []
        self.image_size = image_size
        self.videopath = 'JPEGImages/480p'
        self.maskpath = 'Annotations/480p'
        self.list = os.listdir(root_path+'/JPEGImages/480p/')
        self.skip_list = []


    def __getitem__(self, index):
        ## VALID: IF THE RATE OF MASK IS LESS THAN 0.2, OTHERWISE RESAMPLE
        # valid = False
        while True:
            index = np.random.randint(0,len(self.list))
            if index in self.skip_list:
                continue

            GT_size = self.image_size

            # get GT image
            imgs = self.list[index]
            try:
                Video_GT = read_img(imgs,self.root_path, self.videopath, GT_size)
                Mask_GT, rate = read_img(imgs,self.root_path, self.maskpath, GT_size)
            except Exception:
                raise IOError("Load {} Error".format(imgs))

            if rate<0.2:
                return Video_GT, Mask_GT
            else:
                self.skip_list.append(index)


    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    image_path = 'E:/DAVIS/Annotations/480p/bear/00000.png'
    img_GT = util.read_img(image_path)
    img_GT = util.channel_convert(img_GT.shape[2], 'gray', [img_GT])[0]
    img_GT = cv2.resize(np.copy(img_GT), (256, 256), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB, HWC to CHW, numpy to tensor
    # if img_GT.shape[2] == 3:
    #     img_GT = img_GT[:, :, [2, 1, 0]]
    img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()

    img_GT = torch.where(img_GT>0,torch.ones_like(img_GT),torch.zeros_like(img_GT))
    print(torch.mean(img_GT)) # 0.1225


