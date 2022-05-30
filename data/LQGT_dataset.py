import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# from turbojpeg import TurboJPEG
from PIL import Image
from jpeg2dct.numpy import load, loads
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms.functional as F

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.paths_GT, self.sizes_GT = util.get_image_paths(dataset_opt['dataroot_GT'])

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        scale = self.dataset_opt['scale']
        GT_size = self.dataset_opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]

        img_GT = util.read_img(GT_path)



        img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
        # img_jpeg_GT = util.channel_convert(img_jpeg_GT.shape[2], self.dataset_opt['color'], [img_jpeg_GT])[0]


        ###### directly resize instead of crop
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                            interpolation=cv2.INTER_LINEAR)
        # img_jpeg_GT = cv2.resize(np.copy(img_jpeg_GT), (GT_size, GT_size),
        #                          interpolation=cv2.INTER_LINEAR)


        orig_height, orig_width, _ = img_GT.shape
        H, W, _ = img_GT.shape

        img_gray = rgb2gray(img_GT)
        sigma = 2 #random.randint(1, 4)

        if self.opt['model']=="PAMI" or self.opt['model']=="CLRNet":
            canny_img = canny(img_gray, sigma=sigma, mask=None)
            canny_img = canny_img.astype(np.float)
            canny_img = self.to_tensor(canny_img)
        # elif self.opt['model']=="ICASSP_NOWAY":
        #     canny_img = img_gray
        else:
            canny_img = None


        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            # img_jpeg_GT = img_jpeg_GT[:, :, [2, 1, 0]]
            # img_LQ = img_LQ[:, :, [2, 1, 0]]


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # canny_img = torch.from_numpy(np.ascontiguousarray(canny_img)).float()
        # img_jpeg_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_jpeg_GT, (2, 0, 1)))).float()
        # img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path

        return (img_GT, 0, canny_img if canny_img is not None else img_GT.clone())

    def __len__(self):
        return len(self.paths_GT)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
