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
from attack import *

opt = DefaultConfig()

def test(model):
    original_acc = 0
    adv_acc = 0
    adv_psnr = 0
    adv_l2 = 0
    pic_num = 0
    quality_list = opt.quality_list

    total_acc_list = []
    total_label_list = []
    total_psnr_with_pic_list = []
    total_psnr_with_adv_list = []
    total_l2_with_pic_list = []
    total_l2_with_adv_list = []
    template_list = [0.0 for i in range(0,len(quality_list))]
    name_list = []

    Imagenet = torchvision.datasets.ImageFolder('/home/mengteshi/Imagenet/val')
    I_len = Imagenet.__len__()
    I_class_len = Imagenet.classes.__len__()
    sub_Imagenet = []
    # read in images
    for i in range(pic_num*50,I_len,I_len//I_class_len):
        sub_Imagenet.append((Imagenet[i][0].resize((224, 224)), Imagenet[i][1]))
    # process each image
    for (image, label) in sub_Imagenet:  
        pic_num += 1
        image = torch.FloatTensor(np.asarray(image)).cuda()
        predict_label = torch.argmax(mypredict(model,image),1).item()
        original_acc += (predict_label == label)
        print('-'*20)
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('id:',pic_num)
        print('original label:',label)
        print('predict label:',predict_label)

        if opt.attack_method == 'FGSM' or opt.attack_method == 'IGSM':
            advpic = my_IGSM(model,image,label)
        elif opt.attack_method == 'PGD':
            advpic = PGD(model,image,label)
        else:
            advpic = image

        adv_label = torch.argmax(mypredict(model,advpic),1).item()

        adv_acc += (adv_label == label)
        tmp_psnr = jpgcompress.psnr(image,advpic)
        adv_psnr += tmp_psnr
        tmp_l2 = torch.sqrt(torch.mean((advpic-image)** 2)).item()
        adv_l2 += tmp_l2
        print('adv label:',adv_label)
        print('adv psnr:',tmp_psnr)
        print('adv l2:',tmp_l2)
        print()

        index = -1
        # original image + normal JPEG compress
        index += 1
        total_acc_list.append(template_list[:])
        total_label_list.append(template_list[:])
        total_psnr_with_pic_list.append(template_list[:])
        total_psnr_with_adv_list.append(template_list[:])
        total_l2_with_pic_list.append(template_list[:])
        total_l2_with_adv_list.append(template_list[:])
        name_list.append('jpg pic')
        for i in range(0,len(quality_list)):
            jpgpic=jpgcompress.normalJpegCompress(image,quality_list[i]).float()

            my_label = torch.argmax(mypredict(model,jpgpic),1).item()
            my_psnr_with_pic = jpgcompress.psnr(image,jpgpic)
            my_psnr_with_adv = jpgcompress.psnr(jpgpic,advpic)
            my_l2_with_pic = torch.sqrt(torch.mean((jpgpic-image)** 2)).item()
            my_l2_with_adv = torch.sqrt(torch.mean((jpgpic-advpic)** 2)).item()

            total_acc_list[index][i] += (my_label == label)
            total_label_list[index][i] = my_label
            total_psnr_with_pic_list[index][i] += my_psnr_with_pic
            total_psnr_with_adv_list[index][i] += my_psnr_with_adv
            total_l2_with_pic_list[index][i] += my_l2_with_pic
            total_l2_with_adv_list[index][i] += my_l2_with_adv
        
        # adv + normal JPEG compress
        index += 1
        total_acc_list.append(template_list[:])
        total_label_list.append(template_list[:])
        total_psnr_with_pic_list.append(template_list[:])
        total_psnr_with_adv_list.append(template_list[:])
        total_l2_with_pic_list.append(template_list[:])
        total_l2_with_adv_list.append(template_list[:])
        name_list.append('adv + jpg')
        for i in range(0,len(quality_list)):
            jpgadv=jpgcompress.normalJpegCompress(advpic,quality_list[i]).float()
            
            my_label = torch.argmax(mypredict(model,jpgadv),1).item()
            my_psnr_with_pic = jpgcompress.psnr(image,jpgadv)
            my_psnr_with_adv = jpgcompress.psnr(advpic,jpgadv)
            my_l2_with_pic = torch.sqrt(torch.mean((jpgadv-image)** 2)).item()
            my_l2_with_adv = torch.sqrt(torch.mean((jpgadv-advpic)** 2)).item()

            total_acc_list[index][i] += (my_label == label)
            total_label_list[index][i] = my_label
            total_psnr_with_pic_list[index][i] += my_psnr_with_pic
            total_psnr_with_adv_list[index][i] += my_psnr_with_adv
            total_l2_with_pic_list[index][i] += my_l2_with_pic
            total_l2_with_adv_list[index][i] += my_l2_with_adv

        # adv + Fast adversarial rounding
        if opt.compress_needlist == 1:
            eta_list = opt.eta_list
            for i in range(len(eta_list)):
                total_acc_list.append(template_list[:])
                total_label_list.append(template_list[:])
                total_psnr_with_pic_list.append(template_list[:])
                total_psnr_with_adv_list.append(template_list[:])
                total_l2_with_pic_list.append(template_list[:])
                total_l2_with_adv_list.append(template_list[:])
                name_list.append('Fast adversarial rounding_eta=%s'%eta_list[i]+' jpg adv') 
            
            for i in range(0,len(quality_list)):
                jpgadv_list = jpgcompress.fastAdversarialRounding(image,model,label,quality_list[i],eta_list)

                for jpgadv_i in range(len(jpgadv_list)):
                    my_label = torch.argmax(mypredict(model,jpgadv_list[jpgadv_i]),1).item()
                    my_psnr_with_pic = jpgcompress.psnr(image,jpgadv_list[jpgadv_i])
                    my_psnr_with_adv = jpgcompress.psnr(advpic,jpgadv_list[jpgadv_i])
                    my_l2_with_pic = torch.sqrt(torch.mean((jpgadv_list[jpgadv_i]-image)** 2)).item()
                    my_l2_with_adv = torch.sqrt(torch.mean((jpgadv_list[jpgadv_i]-advpic)** 2)).item()

                    total_acc_list[index+jpgadv_i+1][i] += (my_label == label)
                    total_label_list[index+jpgadv_i+1][i] = my_label
                    total_psnr_with_pic_list[index+jpgadv_i+1][i] += my_psnr_with_pic
                    total_psnr_with_adv_list[index+jpgadv_i+1][i] += my_psnr_with_adv
                    total_l2_with_pic_list[index+jpgadv_i+1][i] += my_l2_with_pic
                    total_l2_with_adv_list[index+jpgadv_i+1][i] += my_l2_with_adv
            
            index += len(eta_list)

        print()
        print("predict accuracy:  %.4f" %(original_acc*1.0/pic_num))
        print("adv accuracy:  %.4f" %(adv_acc*1.0/pic_num))
        print("mean adv psnr:  %.4f" %(adv_psnr*1.0/pic_num))
        print("mean adv l2:  %.4f" %(adv_l2*1.0/pic_num))
        print()
        for kind in range(0,index+1):
            for i in range(0,len(quality_list)):
                print(name_list[kind]+str(quality_list[i])+' label:  %d' %(total_label_list[kind][i]))
                print(name_list[kind]+str(quality_list[i])+' accuracy:  %.4f' %(total_acc_list[kind][i]*1.0/pic_num))
                print(name_list[kind]+str(quality_list[i])+' psnr:  %.4f' %(total_psnr_with_pic_list[kind][i]*1.0/pic_num))
                print(name_list[kind]+str(quality_list[i])+' psnr with adv:  %.4f' %(total_psnr_with_adv_list[kind][i]*1.0/pic_num))
                print(name_list[kind]+str(quality_list[i])+' l2:  %.4f' %(total_l2_with_pic_list[kind][i]*1.0/pic_num))
                print(name_list[kind]+str(quality_list[i])+' l2 with adv:  %.4f' %(total_l2_with_adv_list[kind][i]*1.0/pic_num))
                print()
        print('Finished '+str(pic_num)+'/1000')

def main(): 
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_index
    print(os.getpid())
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    opt.show()
    # import the model
    my_model = opt.my_model.cuda().eval()

    print("\n\nStart Test...")
    test(my_model)
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

if __name__ == '__main__':
    main()