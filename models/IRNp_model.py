import logging
from collections import OrderedDict
import torchvision.transforms.functional_tensor as F_t
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Functional
from noise_layers.salt_pepper_noise import SaltPepper
import torchvision.transforms.functional_pil as F_pil
from skimage.feature import canny
import torchvision
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from skimage.color import rgb2gray
from skimage.metrics._structural_similarity import structural_similarity
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CWLoss
from models.modules.Quantization import Quantization
import torch.distributed as dist
from utils.JPEG import DiffJPEG
from torchvision import models
from loss import AdversarialLoss, PerceptualLoss, StyleLoss
import cv2
from mbrs_models.Encoder_MP import Encoder_MP
from metrics import PSNR, EdgeAccuracy
from .invertible_net import Inveritible_Decolorization_PAMI, ResBlock, DenseBlock
from .crop_localize_net import CropLocalizeNet
from .conditional_jpeg_generator import FBCNN, QF_predictor, MantraNet
from utils import Progbar, create_dir, stitch_images, imsave
import os
import pytorch_ssim
from noise_layers import *
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.middle_filter import MiddleBlur
from noise_layers.resize import Resize
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.crop import Crop
from models.networks import EdgeGenerator, DG_discriminator, InpaintGenerator, Discriminator, NormalGenerator, UNetDiscriminator, \
    JPEGGenerator
from mbrs_models.Decoder import Decoder, Decoder_MLP
# import matlab.engine
from mbrs_models.baluja_networks import HidingNetwork, RevealNetwork
from pycocotools.coco import COCO
from models.conditional_jpeg_generator import domain_generalization_predictor
from loss import ExclusionLoss

# print("Starting MATLAB engine...")
# engine = matlab.engine.start_matlab()
# print("MATLAB engine loaded successful.")
logger = logging.getLogger('base')
# json_path = '/home/qichaoying/Documents/COCOdataset/annotations/incnances_val2017.json'
# load coco data
# coco = COCO(annotation_file=json_path)
#
# # get all image index info
# ids = list(sorted(coco.imgs.keys()))
# print("number of images: {}".format(len(ids)))
#
# # get all coco class labels
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

import lpips

class IRNpModel(BaseModel):
    def __init__(self, opt):
        super(IRNpModel, self).__init__(opt)
        lr_D = 2e-5  # 2*train_opt['lr_G']
        lr_later = 1e-4
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        ########### CONSTANTS ###############
        self.TASK_IMUGEV2 = "ImugeV2"
        self.TASK_TEST = "Test"
        self.TASK_CropLocalize = "CropLocalize"
        self.TASK_RHI3 = "RHI3"
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.task_name = self.train_opt['task_name']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']

        ############## Metrics and attacks #############
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()

        self.jpeg90 = Jpeg(90).cuda()
        self.jpeg80 = Jpeg(80).cuda()
        self.jpeg70 = Jpeg(70).cuda()
        self.jpeg60 = Jpeg(60).cuda()
        self.jpeg50 = Jpeg(50).cuda()
        self.combined_90 = Combined(
            [
                JpegMask(95), Jpeg(95), JpegSS(95), Jpeg(90), JpegMask(90), JpegSS(90),
                JpegMask(85), Jpeg(85), JpegSS(85), Jpeg(80), JpegMask(80), JpegSS(80)
             ]).cuda()
        self.combined_70 = Combined(
            [
                JpegMask(75), Jpeg(75), JpegSS(75), Jpeg(70), JpegMask(70), JpegSS(70),
                JpegMask(65), Jpeg(65), JpegSS(65), Jpeg(60), JpegMask(60), JpegSS(60)
            ]).cuda()
        self.combined_50 = Combined(
            [
                JpegMask(55), Jpeg(55), JpegSS(55), Jpeg(50), JpegMask(50), JpegSS(50),
                JpegMask(45), Jpeg(45), JpegSS(45), Jpeg(40), JpegMask(40), JpegSS(40)
            ]).cuda()
        self.combined_30 = Combined(
            [
                JpegMask(35), Jpeg(35), JpegSS(35), Jpeg(30), JpegMask(30), JpegSS(30),
                JpegMask(25), Jpeg(25), JpegSS(25), Jpeg(20), JpegMask(20), JpegSS(20)
            ]).cuda()
        self.combined_10 = Combined(
            [
                JpegMask(10), Jpeg(10), JpegSS(10),
            ]).cuda()
        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur().cuda()
        self.median_blur = MiddleBlur(kernel=5).cuda()
        self.resize = Resize().cuda()
        self.identity = Identity().cuda()
        self.combined_jpeg_weak = Combined(
            [JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70)]).cuda()
        self.combined_jpeg_medium = Combined(
            [JpegMask(60), Jpeg(60),JpegMask(50), Jpeg(50), JpegMask(40), Jpeg(40)]).cuda()
        self.combined_jpeg_strong = Combined(
            [JpegMask(30), Jpeg(30), JpegMask(20), Jpeg(20), JpegMask(10), Jpeg(10)]).cuda()

        self.bce_loss = nn.BCELoss().cuda()
        self.l1_loss = nn.SmoothL1Loss().cuda()  # reduction="sum"
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
        self.perceptual_loss = PerceptualLoss().cuda()
        self.style_loss = StyleLoss().cuda()
        self.Quantization = Quantization().cuda()
        self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']).cuda()
        self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']).cuda()
        self.criterion_adv = CWLoss().cuda()  # loss for fooling target model
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
        self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()

        ############## Nets ################################
        self.generator = Inveritible_Decolorization_PAMI(dims_in=[[3, 64, 64]], block_num=[2,2,2],subnet_constructor=ResBlock).cuda()
        self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()],
                                                 find_unused_parameters=True)
        
        self.localizer = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, dim=16).cuda()

        # self.discriminator = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        self.discriminator = Discriminator(in_channels=3,use_SRM=False).cuda() #DG_discriminator(in_channels=256,use_SRM=True).cuda()

        self.netG = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[2,2,2],subnet_constructor=ResBlock).cuda()
        
        # self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])
        self.discriminator_mask = Discriminator(in_channels=3,use_SRM=False).cuda()
        self.dis_adv_cov = Discriminator(in_channels=1, use_SRM=False).cuda()

        self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
                                                 find_unused_parameters=True)
        self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[torch.cuda.current_device()],
                                                     find_unused_parameters=True)
        self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                            find_unused_parameters=True)
        self.dis_adv_cov = DistributedDataParallel(self.dis_adv_cov, device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=True)
        self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                          device_ids=[torch.cuda.current_device()],
                                                          find_unused_parameters=True)

        ########### For Crop localization ############
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        ########## Load pre-trained ##################
        # good_models: /models/45010
        self.out_space_storage = '/home/qcying/20220106_IMUGE'
        # self.model_path = '/model/COCO/18010'
        # self.model_path = '/model/CelebA/15010'
        self.model_path = '/model/ILSVRC/2010'

        load_state = True
        if load_state:
            # 45010
            pretrain = self.out_space_storage+self.model_path

            # load_path_G = pretrain + "_domain.pth"
            # if load_path_G is not None:
            #     logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
            #     if os.path.exists(load_path_G):
            #         self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
            #     else:
            #         logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_discriminator.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_discriminator_mask.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator_mask, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            # load_path_G = pretrain + "_dis_adv_cov.pth"
            # if load_path_G is not None:
            #     logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
            #     if os.path.exists(load_path_G):
            #         self.load_network(load_path_G, self.dis_adv_cov, self.opt['path']['strict_load'])
            #     else:
            #         logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_localizer.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.localizer, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))
                    
        self.log_dict = OrderedDict()

        ########## optimizers ##################
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_G = torch.optim.AdamW(optim_params, lr=train_opt['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_G)

        # for domain generator
        optim_params = []
        for k, v in self.generator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_generator = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                             weight_decay=wd_G,
                                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_generator)

        # for mask discriminator
        optim_params = []
        for k, v in self.discriminator_mask.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_discriminator_mask = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                             weight_decay=wd_G,
                                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_discriminator_mask)

        # for mask discriminator
        optim_params = []
        for k, v in self.dis_adv_cov.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_dis_adv_cov = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                             weight_decay=wd_G,
                                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_dis_adv_cov)

        # for discriminator
        optim_params = []
        for k, v in self.discriminator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_discriminator = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                             weight_decay=wd_G,
                                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_discriminator)

        # localizer
        optim_params = []
        for k, v in self.localizer.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_localizer = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                    weight_decay=wd_G,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_localizer)
       
        # ############## schedulers #########################
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state']))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        ######## init constants
        self.forward_image_buff = None

    def feed_data(self, batch):
        # if self.train_opt['using_self_defined_dataset'] == 1.0:
        #     # {'GT': img_GT, 'GT_path': GT_path}
        #     data, label = batch
        #     self.real_H = data['GT'].cuda()
        #     self.jpeg_real_H = data['JPEG'].cuda()
        #     self.QF_GT = data['QF'].type(torch.FloatTensor).cuda()
        #     self.real_H_path = data['GT_path']
        #     # self.label_GT = label.type(torch.FloatTensor).cuda().unsqueeze(1) # regression
        #     self.label_GT = label.cuda()
        # else:
        img, label, canny_image = batch
        self.real_H = img.cuda()
        self.canny_image = canny_image.cuda()

        # self.ref_L = data['LQ'].cuda()  # LQ
        # self.real_H = data['GT'].cuda()  # GT

    def gaussian_batch(self, dims):
        return torch.clamp(torch.randn(tuple(dims)).cuda(), 0, 1)

    def loss_forward(self, label_GT, label_pred, out, y, z):
        is_targeted = False
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z ** 2) / z.shape[0]
        loss_adv = None
        if label_GT is not None:
            loss_adv = 2 * self.criterion_adv(label_pred, label_GT, is_targeted)

        return l_forw_fit, l_forw_ce, loss_adv

    def loss_backward(self, label_pred, label_GT, GT_ref, reversed_image):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(GT_ref, reversed_image)

        # loss_label = self.criterion(label_pred, label_GT)
        loss_label = None
        return l_back_rec, loss_label

    def loss_forward_and_backward_imuge(self, fake_outputs, cover_images, masks=None, use_l1=True, use_vgg=False,use_percept=False):
        gen_loss = 0
        if use_l1:
            gen_l1_loss = self.l1_loss(fake_outputs, cover_images)
        else:
            gen_l1_loss = self.l2_loss(fake_outputs, cover_images)
        gen_loss += gen_l1_loss
        gen_l1_local_loss = None
        if masks is not None:
            if use_l1:
                gen_l1_local_loss = self.l1_loss(fake_outputs * masks,
                                                 cover_images * masks) / torch.mean(masks)
            else:
                gen_l1_local_loss = self.l2_loss(fake_outputs * masks,
                                                 cover_images * masks) / torch.mean(masks)
            # gen_loss += 2 * gen_l1_local_loss

        if use_percept and cover_images.shape[1] == 3:
            # generator perceptual loss

            gen_content_loss = self.perceptual_loss(fake_outputs, cover_images)
            gen_loss += 0.01 * gen_content_loss

        if use_vgg and cover_images.shape[1] == 3:
            l_forward_ssim = - self.ssim_loss(fake_outputs, cover_images)
            gen_loss += 0.01 * l_forward_ssim

        # generator style loss
        # if masks is not None:
        #     gen_style_loss = self.style_loss(fake_outputs, cover_images)
        #     gen_style_loss = gen_style_loss * 250
        #     gen_loss += gen_style_loss

        return gen_l1_local_loss, gen_l1_loss, gen_loss

    def symm_pad(self, im, padding):
        h, w = im.shape[-2:]
        left, right, top, bottom = padding

        x_idx = np.arange(-left, w + right)
        y_idx = np.arange(-top, h + bottom)

        x_pad = self.reflect(x_idx, -0.5, w - 0.5)
        y_pad = self.reflect(y_idx, -0.5, h - 0.5)
        xx, yy = np.meshgrid(x_pad, y_pad)
        return im[..., yy, xx]

    def reflect(self, x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    def optimize_parameters(self, step, latest_values=None, train=True, eval_dir=None):
        self.netG.train()
        self.localizer.train()
        self.discriminator.train()
        self.discriminator_mask.train()
        self.global_step = self.global_step + 1
        logs, debug_logs = [], []
      
        self.real_H = torch.clamp(self.real_H, 0, 1)
        batch_size = self.real_H.shape[0]

        save_interval = 1000
        only_gen_immunized = self.opt["only_gen_immunized"]
        if only_gen_immunized == 1.0:
            with torch.no_grad():
                forward_stuff = self.netG(x=torch.cat((self.real_H, self.canny_image), dim=1))
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                forward_image = torch.clamp(forward_image, 0, 1)
                forward_null = torch.clamp(forward_null, 0, 1)

                # ####### Save independent images #############
                main_folder = self.out_space_storage + '/ori_{}_0114'.format(self.opt["datasets"]["train"]["name"])
                if not os.path.exists(main_folder): os.mkdir(main_folder)
                name =  main_folder + "/"+ str(self.global_step).zfill(5) + "_" + str(
                    self.rank) + ".png"
                for image_no in range(self.real_H.shape[0]):
                    camera_ready = self.real_H[image_no].unsqueeze(0)
                    torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                 name, nrow=1, padding=0,
                                                 normalize=False)
                print("Saved:{}".format(name))

                main_folder = self.out_space_storage + '/immu_{}_0114'.format(self.opt["datasets"]["train"]["name"])
                if not os.path.exists(main_folder): os.mkdir(main_folder)
                name = main_folder + "/"+ str(self.global_step).zfill(5) + "_" + str(
                    self.rank) + ".png"
                for image_no in range(forward_image.shape[0]):
                    camera_ready = forward_image[image_no].unsqueeze(0)
                    torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                 name, nrow=1, padding=0,
                                                 normalize=False)
                print("Saved:{}".format(name))
                return logs, debug_logs

        with torch.enable_grad():
            is_real_train = True
            ######################### iMUGE ###########################################################
            masks_GT = torch.zeros(batch_size, 1, self.real_H.shape[2], self.real_H.shape[3]).cuda()
            self.tamper_rate = 0.2
            is_data_augment = np.random.rand() < 0.0
            is_copy_move = not is_data_augment and self.global_step % 3==1
            percent_range = (0.00, 0.4) #if not is_copy_move else (0.0, 0.2)
            for imgs in range(batch_size):
                masks_origin, self.tamper_rate = self.generate_stroke_mask(
                    [self.real_H.shape[2], self.real_H.shape[3]],percent_range=percent_range)
                masks_origin = masks_origin.cuda()
                masks_GT[imgs, :, :, :] = masks_origin

            masks = masks_GT.repeat(1, 3, 1, 1)

            if self.previous_images is not None:
                if is_data_augment:
                    modified_input = (self.real_H * (1 - masks) + self.previous_images * masks).clone().detach()
                    modified_input = torch.clamp(modified_input, 0, 1)

                    for icn in range(batch_size):
                        img_GT = self.tensor_to_image(modified_input[icn, :, :, :])
                        img_gray = rgb2gray(img_GT)
                        sigma = 2  # random.randint(1, 4)
                        cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                        cannied = self.image_to_tensor(cannied).cuda()
                        self.canny_image[icn, :, :, :] = cannied

                else:
                    modified_input = self.real_H
                # print(self.canny_image.shape)
                forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                forward_image = torch.clamp(forward_image, 0, 1)
                forward_null = torch.clamp(forward_null, 0, 1)
                # forward_image = self.Quantization(forward_image)

                self.forward_image_buff = forward_image.clone().detach()

            if self.global_step>-1 and self.previous_images is not None and self.previous_previous_images is not None:
                # We use previous image as important content in the original image to prevent the networks from refering to the image prior
                # Then, we use previous previous image to tamper the image in the same location
                # l_atk, l_qf, pQF = self.train_JPEG_generator()

                # ####### Save independent images #############
                save_independent = False
                if save_independent:
                    name = self.out_space_storage + '/ori_{}_0114/'.format(self.opt["datasets"]["train"]["name"]) + str(self.global_step).zfill(5) + "_" + str(
                        self.rank) + ".png"
                    for image_no in range(self.real_H.shape[0]):
                        camera_ready = self.real_H[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = self.out_space_storage + '/immu_{}_0114/'.format(self.opt["datasets"]["train"]["name"]) + str(self.global_step).zfill(5) + "_" + str(
                        self.rank) + ".png"
                    for image_no in range(forward_image.shape[0]):
                        camera_ready = forward_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                ####### Tamper ###############
                # attacked_forward = forward_image * (1 - masks) + self.previous_previous_images * masks
                if is_copy_move:
                    # attacked_forward = self.splicing(forward_image, self.previous_previous_images, masks)

                    # copy_move should at least
                    tamper = forward_image.clone().detach()

                    x_shift = int((self.real_H.shape[2]) * (np.random.rand() - 0.5))
                    y_shift = int((self.real_H.shape[3]) * (np.random.rand() - 0.5))

                    ### two times padding ###
                    self.mask_shifted = torch.zeros((masks.shape[0], masks.shape[1],
                                                masks.shape[2] + abs(2 * x_shift),
                                                masks.shape[3] + abs(2 * y_shift))).cuda()
                    self.tamper_shifted = torch.zeros((self.real_H.shape[0], self.real_H.shape[1],
                                                  self.real_H.shape[2] + abs(2 * x_shift),
                                                  self.real_H.shape[3] + abs(2 * y_shift))).cuda()
                    self.mask_shifted[:, :,
                    abs(x_shift) + x_shift:abs(x_shift) + x_shift + self.real_H.shape[2],
                    abs(y_shift) + y_shift:abs(y_shift) + y_shift + self.real_H.shape[3]] = masks
                    self.tamper_shifted[:, :, abs(x_shift) + x_shift: abs(x_shift) + x_shift + self.real_H.shape[2],
                    abs(y_shift) + y_shift: abs(y_shift) + y_shift + self.real_H.shape[3]] = tamper

                    self.mask_shifted = self.mask_shifted[:, :,
                                   abs(x_shift):abs(x_shift) + self.real_H.shape[2],
                                   abs(y_shift):abs(y_shift) + self.real_H.shape[3]]
                    self.tamper_shifted = self.tamper_shifted[:, :,
                                     abs(x_shift): abs(x_shift) + self.real_H.shape[2],
                                     abs(y_shift): abs(y_shift) + self.real_H.shape[3]]

                    masks = self.mask_shifted.clone().detach()
                    masks = torch.clamp(masks, 0, 1)
                    valid = torch.mean(masks)

                    masks_GT = masks[:, :1, :, :]
                    attacked_forward = forward_image * (1 - masks) + self.tamper_shifted.clone().detach() * masks
                    del self.tamper_shifted
                    del self.mask_shifted
                    attack_name="copymove"
                else:
                    attacked_forward = forward_image * (1 - masks) + self.previous_previous_images * masks
                    attack_name = "splicing"
                logs.append(('Kind',attack_name))
                attacked_forward = torch.clamp(attacked_forward, 0, 1)
                attack_full_name = ""

                # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                # if self.global_step % 4 < 2:
                # attacked_image = torch.zeros((4,forward_image.shape[1],forward_image.shape[2],forward_image.shape[3]),dtype=torch.float32).cuda()
                way_attack = 8
                num_attacks = way_attack * batch_size
                # attacked_image = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),dtype=torch.float32).cuda()

                attacked_image_0 = self.resize(attacked_forward)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack:imgs * way_attack + 1, :, :, :] = attacked_image_0[imgs:imgs + 1, :, :, :]
                # attack_layer = self.combined_jpeg_weak
                attack_layer_1 = self.combined_jpeg_weak
                # attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                # beta = np.random.rand()
                # attacked_image_1 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = attacked_forward_1 #torch.clamp(attacked_image_1, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+1:imgs * way_attack + 2, :, :, :] = attacked_image_1[imgs:imgs + 1, :, :, :]
                #     attack_full_name += "JPEG"
                # elif self.global_step % 4 < 3:
                attacked_image_2 = self.median_blur(attacked_forward)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+2:imgs * way_attack + 3, :, :, :] = attacked_image_2[imgs:imgs + 1, :, :, :]
                # attack_full_name += "MBlur"
                # else: #if self.global_step % 4 < 2:
                attacked_image_3 = self.gaussian_blur(attacked_forward)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+3:imgs * way_attack + 4, :, :, :] = attacked_image_3[imgs:imgs + 1, :, :, :]
                # attack_layer = self.combined_jpeg_medium
                attack_layer_1 = self.combined_jpeg_medium
                # attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                # beta = np.random.rand()
                # attacked_image_4 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_4 = attacked_forward_1 #torch.clamp(attacked_image_4, 0, 1)

                attack_layer = self.combined_jpeg_strong
                attack_layer_1 = self.combined_jpeg_strong
                # attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                # beta = np.random.rand()
                # attacked_image_5 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_5 = attacked_forward_1 #torch.clamp(attacked_image_5, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+5:imgs * way_attack + 6, :, :, :] = attacked_image_5[imgs:imgs + 1, :, :, :]

                attacked_image_6 = self.gaussian(attacked_forward)
                attacked_image_6 = torch.clamp(attacked_image_6, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+6:imgs * way_attack + 7, :, :, :] = attacked_image_6[imgs:imgs + 1, :, :, :]

                attacked_image_7 = self.identity(attacked_forward)
                attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)
                # for imgs in range(batch_size):
                #     attacked_image[imgs * way_attack+7:imgs * way_attack + 8, :, :, :] = attacked_image_7[imgs:imgs + 1, :, :, :]

                attack_full_name += "Mixed"
                attacked_forward_expanded = attacked_forward.repeat(way_attack, 1, 1, 1)
                simulated_images = torch.cat((attacked_image_0,attacked_image_1,attacked_image_2,attacked_image_3
                                              ,attacked_image_4,attacked_image_5,
                                            attacked_image_6,attacked_image_7),dim=0)

                diff_images = simulated_images - attacked_forward_expanded
                attacked_image = attacked_forward_expanded + diff_images.clone().detach()
                attacked_image = self.Quantization(attacked_image)

                logs.append(('Kind', attack_full_name))
                masks_expand = masks.repeat(way_attack,1,1,1)
                masks_GT_expand = masks.repeat(way_attack, 1, 1, 1)
                # print(masks_expand.shape)
                tampered_attacked_image = attacked_image * (1 - masks_expand)
                rectified_crop_padded_image = torch.clamp(tampered_attacked_image, 0, 1)

                forward_expand = forward_image.repeat(way_attack, 1, 1, 1)
                canny_expanded = self.canny_image.repeat(way_attack,1,1,1)
                masks_GT_expand = masks_GT.repeat(way_attack,1,1,1)
                modified_expand = modified_input.repeat(way_attack,1,1,1)


                # canny_input = torch.zeros_like(canny_expanded).cuda()
                # for icn in range(batch_size):
                #     img_GT = self.tensor_to_image(rectified_crop_padded_image[icn,:,:,:])
                #     img_gray = rgb2gray(img_GT)
                #     sigma = 2  # random.randint(1, 4)
                #     cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                #     cannied = self.image_to_tensor(cannied).cuda()
                #     canny_input[icn,:,:,:] = cannied
                canny_input = (torch.zeros_like(self.canny_image).cuda())
                canny_input = canny_input.repeat(way_attack, 1, 1, 1)

                reversed_stuff, reverse_feature = self.netG(torch.cat((rectified_crop_padded_image,canny_input),dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:,:3,:,:], reversed_stuff[:,3:,:,:]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                reversed_image = reversed_ch1
                reversed_canny = reversed_ch2

                # #### domain generalization discriminator ###
                # self.generator.train()
                # dg_pred = self.generator(reverse_feature.clone().detach())
                # l_dg = self.l2_loss(dg_pred,attack_rate_tensor)
                # self.optimizer_generator.zero_grad()
                # l_dg.backward()
                # self.optimizer_generator.step()
                lr = self.get_current_learning_rate()
                logs.append(('lr', lr))
                # logs.append(('DG', l_dg.item()))
                # dg_pred = self.generator(reverse_feature)
                #
                # l_dg = - self.l2_loss(dg_pred, attack_rate_tensor)

#################### DISCRIMINATORS ##################################################
                ########### discriminator loss for marked image
                # dis_input_real = modified_input
                # dis_input_fake = forward_image.detach()
                # dis_real = self.discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
                # dis_fake = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
                # dis_real_loss = self.bce_loss(dis_real,torch.ones_like(dis_real))
                # dis_fake_loss = self.bce_loss(dis_fake,torch.zeros_like(dis_fake))
                # dis_loss = (dis_real_loss + dis_fake_loss) / 2
                # self.discriminator.train()
                # self.optimizer_discriminator.zero_grad()
                # dis_loss.backward()
                # if self.train_opt['gradient_clipping']:
                #     nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_opt['gradient_clipping'])
                # self.optimizer_discriminator.step()
                # ########## generator adversarial loss
                # gen_input_fake = forward_image
                # gen_fake = self.discriminator(gen_input_fake)
                # FW_GAN = self.bce_loss(gen_fake,torch.ones_like(dis_real))
                # logs.append(('FW_GAN', FW_GAN.item()))
                # # gen_marked_feat = 0
                # # for i in range(len(dis_real_feat)):
                # #     gen_marked_feat += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
                # # gen_marked_loss = gen_marked_feat * 5
                # # FW_GAN += gen_marked_loss
                # #
                # ########### discriminator loss for recovered image
                # dis_input_real = modified_expand
                # dis_input_fake = reversed_image.detach()
                # dis_real = self.discriminator_mask(dis_input_real)  # in: (grayscale(1) + edge(1))
                # dis_fake = self.discriminator_mask(dis_input_fake)  # in: (grayscale(1) + edge(1))
                # dis_real_loss = self.bce_loss(dis_real, torch.ones_like(dis_real))
                # downsample_mask = Functional.interpolate(
                #                     1-masks_GT_expand,
                #                     size=[dis_fake.shape[2], dis_fake.shape[3]],
                #                     mode='bilinear')
                # downsample_mask = torch.clamp(downsample_mask,0,1)
                # dis_fake_loss = self.bce_loss(dis_fake,downsample_mask)
                # # dis_fake_loss = self.bce_loss(dis_real,torch.zeros_like(dis_real))
                # dis_loss = (dis_real_loss + dis_fake_loss) / 2
                # self.discriminator_mask.train()
                # self.optimizer_discriminator_mask.zero_grad()
                # dis_loss.backward()
                # if self.train_opt['gradient_clipping']:
                #     nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), self.train_opt['gradient_clipping'])
                # self.optimizer_discriminator_mask.step()
                # ########## generator adversarial loss
                # gen_input_fake = reversed_image
                # gen_fake = self.discriminator_mask(gen_input_fake)
                # REV_GAN = self.bce_loss(gen_fake, torch.ones_like(dis_real)) #/ torch.mean(masks)
                # logs.append(('REV_GAN', REV_GAN.item()))
                # # gen_style_loss = 0
                # # for i in range(len(dis_real_feat)):
                # #     gen_style_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
                # # gen_style_loss = gen_style_loss * 10
                # # REV_GAN += gen_style_loss
                #
                # #### discrim for edge
                # dis_input_real = self.canny_image
                # dis_input_fake = reversed_canny.detach()
                # dis_real, dis_real_feat = self.dis_adv_cov(dis_input_real)  # in: (grayscale(1) + edge(1))
                # dis_fake, _ = self.dis_adv_cov(dis_input_fake)  # in: (grayscale(1) + edge(1))
                # dis_real_loss = self.bce_loss(dis_real, torch.ones_like(dis_real))
                # downsample_mask = Functional.interpolate(
                #     1 - masks_GT,
                #     size=[dis_fake.shape[2], dis_fake.shape[3]],
                #     mode='bilinear')
                # downsample_mask = torch.clamp(downsample_mask, 0, 1)
                # dis_fake_loss = self.bce_loss(dis_fake, downsample_mask)
                # dis_loss = (dis_real_loss + dis_fake_loss) / 2
                # self.dis_adv_cov.train()
                # self.optimizer_dis_adv_cov.zero_grad()
                # dis_loss.backward()
                # if self.train_opt['gradient_clipping']:
                #     nn.utils.clip_grad_norm_(self.dis_adv_cov.parameters(), se3lf.train_opt['gradient_clipping'])
                # self.optimizer_dis_adv_cov.step()
                # ########## generator adversarial loss
                # gen_input_fake = reversed_canny
                # gen_fake, gen_fake_feat = self.dis_adv_cov(gen_input_fake)
                # REVedge_GAN = self.bce_loss(gen_fake, torch.ones_like(dis_real))  # / torch.mean(masks)
                # logs.append(('REVedge_GAN', REVedge_GAN.item()))
                # gen_fm_loss = 0
                # for i in range(len(dis_real_feat)):
                #     gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
                # gen_fm_loss = gen_fm_loss * 10
                # REVedge_GAN += gen_fm_loss


                ######## Train Localizer for recover ##########
                diffused_image = attacked_image.clone().detach()

                localizer_input = diffused_image
                gen_input_fake = localizer_input
                gen_fake, _ = self.localizer(gen_input_fake)
                CE = self.bce_loss(gen_fake, masks_GT_expand)
                self.localizer.train()
                self.optimizer_localizer.zero_grad()
                CE.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                self.optimizer_localizer.step()
                logs.append(('CE', CE.item()))

                gen_input_fake = attacked_image
                self.optimizer_localizer.zero_grad()
                gen_fake, _ = self.localizer(gen_input_fake)
                CE = self.bce_loss(gen_fake, masks_GT_expand)

                # ########### domain discriminator differentiates jpeg with blur
                # print(reverse_feature.shape)
                # dis_input_fake = reverse_feature.detach()
                # dis_fake, _ = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
                # dis_fake_loss = self.bce_loss(dis_fake,torch.zeros_like(dis_fake) if self.global_step%4<2 else torch.ones_like(dis_fake))
                # self.discriminator.train()
                # self.optimizer_discriminator.zero_grad()
                # dis_fake_loss.backward()
                # if self.train_opt['gradient_clipping']:
                #     nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_opt['gradient_clipping'])
                # self.optimizer_discriminator.step()
                #
                # self.optimizer_discriminator.zero_grad()
                # dis_fake, _ = self.discriminator(dis_input_fake)
                # l_dg = self.bce_loss(dis_fake, torch.ones_like(dis_fake) if self.global_step % 4 < 2 else torch.zeros_like(dis_fake))
                # logs.append(('l_dg', l_dg.item()))

                # loss
                # _, l_forward_l1, l_forward = self.loss_forward_and_backward_imuge(forward_image, modified_input, masks=None, use_l1=True)
                # l_backward_l1_local, l_backward_l1, l_backward = self.loss_forward_and_backward_imuge(reversed_image, modified_input, masks=masks, use_l1=True)
                distance = self.l1_loss # if np.random.rand()>0.7 else self.l2_loss
                l_forward = distance(forward_image, modified_input)
                # l_forward += distance(forward_null, self.canny_image)

                l_backward = distance(reversed_image, modified_expand)
                # l_backward = distance(reversed_image, forward_image.clone().detach())
                l_back_canny = distance(reversed_canny, canny_expanded)
                l_backward += l_back_canny
                l_backward_l1_local = (distance(reversed_image * masks_expand, modified_expand * masks_expand)) / (1e-3+torch.mean(masks))
                # l_backward_l1_local = (distance(reversed_image * masks, forward_image.clone().detach() * masks)) / torch.mean(masks)

                logs.append(('lF', l_forward.item()))
                logs.append(('lB', l_backward.item()))
                logs.append(('lBedge', l_back_canny.item()))
                logs.append(('local', l_backward_l1_local.item()))
                alpha_forw, alpha_back, gamma, delta = 1.0, 0.75, 0.1, 0.01

                psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()
                psnr_backward = self.psnr(self.postprocess(modified_expand), self.postprocess(reversed_image)).item()
                # psnr_comp = self.psnr(self.postprocess(forward_image), self.postprocess(reversed_image)).item()
                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))
                # logs.append(('Pad', psnr_comp))

                # nullify
                l_null = (self.l2_loss(forward_null,torch.zeros_like(self.canny_image).cuda()))
                logs.append(('NL', l_null.item()))
                l_forward += 8*l_null
                psnr_thresh = 35
                loss = 0
                loss += (3 if psnr_forward<psnr_thresh else alpha_forw) * l_forward
                loss += alpha_back * (l_backward+ (3 if torch.mean(masks_GT_expand)>0.2 else 1)*l_backward_l1_local)
                l_backward += alpha_back * (3 if psnr_forward-psnr_backward>1 else 1) * (distance(reversed_canny * masks_GT_expand, canny_expanded * masks_GT_expand)) / (
                            1e-3 + torch.mean(masks_GT_expand))

                loss += 0.01 * CE
                #        + alpha_back * l_bac
                #       kward_l1_local
                # loss += delta * FW_GAN
                # loss += delta * (REV_GAN)
                # loss += delta * REVedge_GAN
                # loss += 1.0 * l_null
                # loss += 1 * l_dg

                # print(l_forward)
                # print(l_backward)
                # print(l_backward_l1_local)
                # print(REV_GAN)
                # print(CE)
                # if psnr_forward<28:

                #     # generator perceptual loss
                # l_percept_fw_vgg = self.perceptual_loss(forward_image, modified_input)
                # l_percept_bk_vgg = self.perceptual_loss(reversed_image, modified_input)

                l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                l_percept_bk_ssim = - self.ssim_loss(reversed_image, modified_expand)
                # # l_percept_canny_ssim = - self.ssim_loss(reversed_canny, canny_expanded)
                loss += (0.3 if psnr_forward<psnr_thresh else 0.1) * l_percept_fw_ssim
                loss += 0.1 * (l_percept_bk_ssim)
                # loss += delta * (l_percept_canny_ssim)
                # loss += delta * l_dg

                # # l_percept_fw_lpips = torch.mean(self.lpips_vgg.forward(forward_image, modified_input))
                # # l_percept_bk_lpips = torch.mean(self.lpips_vgg.forward(reversed_image, modified_input))
                # l_percept_fw = (l_percept_fw_vgg + l_percept_fw_ssim )/2
                # l_percept_bk = (l_percept_bk_vgg + l_percept_bk_ssim )/2
                # logs.append(('lpF', l_percept_fw.item()))
                # logs.append(('lpB', l_percept_bk.item()))
                # logs.append(('lssimF', l_percept_fw_ssim.item()))
                # # logs.append(('lpipF', l_percept_fw_lpips.item()))
                # logs.append(('lvggF', l_percept_fw_vgg.item()))
                # logs.append(('lssimB', l_percept_bk_ssim.item()))
                # # logs.append(('lpipF', l_percept_fw_lpips.item()))
                # logs.append(('lvggB', l_percept_bk_vgg.item()))
                # loss +=  1.5 * l_percept_fw
                # loss += 0.1 * CE
                # loss += 0.1 * FW_GAN
                # if psnr_forward>20:
                #     loss += 0.1 * (REV_GAN)
                #
                #     loss += 1 * l_percept_bk
                #     loss += 2 * l_backward_l1_local

                    # l_forward_ssim = - self.ssim_loss(fake_outputs, cover_images)
                    # gen_loss += 0.01 * l_forward_ssim

                self.netG.train()
                self.optimizer_G.zero_grad()
                loss.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                self.optimizer_G.step()

                SSFW = (-l_percept_fw_ssim).item()
                SSBK = (-l_percept_bk_ssim).item()
                logs.append(('SF', SSFW))
                logs.append(('SB', SSBK))
                # logs.append(('pred', dg_pred.detach().cpu().numpy()))
                # logs.append(('gt', attack_rate_tensor.detach().cpu().numpy()))
                logs.append(('FW', psnr_forward))
                logs.append(('BK', psnr_backward))
                logs.append(('L-RealTime', l_backward_l1_local.item()))

                if step % 200 == 10:
                    images = stitch_images(
                        self.postprocess(modified_expand),
                        self.postprocess(canny_expanded),
                        self.postprocess(forward_expand),
                        self.postprocess(10 * torch.abs(modified_expand - forward_expand)),
                        # self.postprocess(predicted_marked),
                        self.postprocess(diffused_image),
                        # self.postprocess(rectified_crop_padded_image),
                        self.postprocess(masks_GT_expand),
                        self.postprocess(gen_fake),
                        self.postprocess(10 * torch.abs(masks_GT_expand - gen_fake)),
                        self.postprocess(reversed_image),
                        self.postprocess(10 * torch.abs(modified_expand - reversed_image)),
                        self.postprocess(reversed_canny),
                        self.postprocess(10 * torch.abs(canny_expanded - reversed_canny)),

                        # self.postprocess(gen_fake_1),
                        # self.postprocess(gen_fake_mask_1),
                        # self.postprocess(gen_fake_mask_2),

                        img_per_row=1
                    )



                    name = self.out_space_storage + '/images/'+self.opt['datasets']['train']['name']+'/'+str(self.global_step).zfill(5) + "_ "+str(self.rank) + ".png"
                    print('\nsaving sample ' + name)
                    images.save(name)


        ######## Finally ####################
        if step % save_interval== 10 and self.rank <= 0:

            logger.info('Saving models and training states.')
            self.save(self.global_step)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            # former: splicing, later: copy-move
            self.previous_images = self.real_H
                # if self.forward_image_buff is None else \
                # self.forward_image_buff.clone().detach() if np.random.rand()<0.5 else self.real_H.clone().detach()
        return logs, debug_logs

    def splicing(self,source,tamper, mask):
        return source * (1 - mask) + tamper * mask

    def copy_move(self,source,mask):
        tamper = source.clone().detach()

        x_shift = int((self.real_H.shape[2]) / 4 * (np.random.rand()-0.5))
        y_shift = int((self.real_H.shape[3]) / 4 * (np.random.rand() - 0.5))


        ### two times padding ###
        mask_shifted = torch.zeros((mask.shape[0],mask.shape[1],
                                    mask.shape[2]+abs(2*x_shift),
                                    mask.shape[3]+abs(2*y_shift))).cuda()
        tamper_shifted = torch.zeros((self.real_H.shape[0], self.real_H.shape[1],
                                    self.real_H.shape[2] + abs(2 * x_shift),
                                      self.real_H.shape[3] + abs(2 * y_shift))).cuda()
        mask_shifted[:,:,
        abs(x_shift)+x_shift:abs(x_shift)+x_shift+self.real_H.shape[2],
        abs(y_shift)+y_shift:abs(y_shift)+y_shift+self.real_H.shape[3]] = mask
        tamper_shifted[:, :, abs(x_shift) + x_shift: abs(x_shift) + x_shift + self.real_H.shape[2],
                            abs(y_shift) + y_shift: abs(y_shift) + y_shift + self.real_H.shape[3]] = tamper

        mask_shifted = mask_shifted[:,:,
                       abs(x_shift):abs(x_shift)+self.real_H.shape[2],
                       abs(y_shift):abs(y_shift)+self.real_H.shape[3]]
        tamper_shifted = tamper_shifted[:, :,
                         abs(x_shift): abs(x_shift) + self.real_H.shape[2],
                         abs(y_shift): abs(y_shift) + self.real_H.shape[3]]

        mask_shifted = mask_shifted.clone().detach()
        source = source*(1-mask_shifted)+tamper_shifted.clone().detach() * mask_shifted

        return source, mask_shifted

    def is_image_file(self,filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)


    def get_paths_from_images(self,path):
        '''get image path list from image folder'''
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    # img_path = os.path.join(dirpath, fname)
                    images.append((path, dirpath[len(path) + 1:], fname))
        assert images, '{:s} has no valid image file'.format(path)
        return images

    def evaluate(self,data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
        # self.netG.eval()
        # self.localizer.eval()
        with torch.no_grad():
            psnr_forward_sum, psnr_backward_sum = [0,0,0,0,0],  [0,0,0,0,0]
            ssim_forward_sum, ssim_backward_sum =  [0,0,0,0,0],  [0,0,0,0,0]
            F1_sum =  [0,0,0,0,0]
            valid_images = [0,0,0,0,0]
            logs, debug_logs = [], []
            image_list_origin = None if data_origin is None else self.get_paths_from_images(data_origin)
            image_list_immunize = None if data_immunize is None else self.get_paths_from_images(data_immunize)
            image_list_tamper = None if data_tampered is None else self.get_paths_from_images(data_tampered)
            image_list_tampersource = None if data_tampersource is None else self.get_paths_from_images(data_tampersource)
            image_list_mask = None if data_mask is None else self.get_paths_from_images(data_mask)

            for idx in range(len(image_list_origin)):

                p, q, r = image_list_origin[idx]
                ori_path = os.path.join(p, q, r)
                img_GT = self.load_image(ori_path)
                print("Ori: {} {}".format(ori_path, img_GT.shape))
                self.real_H = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.real_H = torch.clamp(self.real_H, 0, 1)
                img_gray = rgb2gray(img_GT)
                sigma = 2  # random.randint(1, 4)
                cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                self.canny_image = self.image_to_tensor(cannied).cuda().unsqueeze(0)
                
                p, q, r = image_list_immunize[idx]
                immu_path = os.path.join(p, q, r)
                img_GT = self.load_image(immu_path)
                print("Imu: {} {}".format(immu_path, img_GT.shape))
                self.immunize = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.immunize = torch.clamp(self.immunize, 0, 1)
                p, q, r = image_list_tamper[idx]
                attack_path = os.path.join(p, q, r)
                img_GT = self.load_image(attack_path)
                print("Atk: {} {}".format(attack_path, img_GT.shape))
                self.attacked_image = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.attacked_image = torch.clamp(self.attacked_image, 0, 1)
                p, q, r = image_list_tampersource[idx]
                another_path = os.path.join(p, q, r)
                img_GT = self.load_image(another_path)
                print("Another: {} {}".format(another_path, img_GT.shape))
                self.another_image = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.another_image = torch.clamp(self.another_image, 0, 1)
                p, q, r = image_list_mask[idx]
                mask_path = os.path.join(p, q, r)
                img_GT = self.load_image(mask_path, grayscale=True)
                print("Mask: {} {}".format(mask_path, img_GT.shape))
                self.mask = self.img_random_crop(img_GT, 512, 512, grayscale=True).cuda().unsqueeze(0)
                self.mask = torch.clamp(self.mask, 0, 1)
                self.mask = self.mask.repeat(1,3,1,1)

                ### skip images with too large tamper masks
                masked_rate = torch.mean(self.mask)
                redo_gen_mask = masked_rate>0.5
                    # print("Masked rate exceed maximum: {}".format(masked_rate))
                    # continue

                catogory = min(4,int(masked_rate*20))
                valid_images[catogory] += 1
                is_copy_move = False
                if True: #self.immunize is None:
                    ##### re-generates immunized images ########
                    modified_input = self.real_H
                    # print(self.canny_image.shape)
                    forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))
                    self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                    self.immunize = torch.clamp(self.immunize, 0, 1)
                    self.immunize = self.Quantization(self.immunize)
                    forward_null = torch.clamp(forward_null, 0, 1)

                ####### Tamper ###############
                if True: #self.attacked_image is None:

                    self.attacked_image = self.immunize * (1-self.mask) + self.another_image * self.mask
                    self.attacked_image = torch.clamp(self.attacked_image, 0, 1)

                # else:
                #     valid = 0
                #     while valid<0.05:
                #         tamper = self.immunize.clone().detach()
                #         masks_origin, self.tamper_rate = self.generate_stroke_mask(
                #             [self.real_H.shape[2], self.real_H.shape[3]], percent_range=(0.1,0.4))
                #         masks_origin = masks_origin.cuda()
                #         self.mask = masks_origin.repeat(1, 3, 1, 1)
                #
                #         x_shift = int((self.real_H.shape[2]) * (2 * np.random.rand() - 1))
                #         y_shift = int((self.real_H.shape[3]) * (2 * np.random.rand() - 1))
                #
                #         ### two times padding ###
                #         self.mask_shifted = torch.zeros((self.mask.shape[0], self.mask.shape[1],
                #                                          self.mask.shape[2] + abs(2 * x_shift),
                #                                          self.mask.shape[3] + abs(2 * y_shift))).cuda()
                #         self.tamper_shifted = torch.zeros((self.real_H.shape[0], self.real_H.shape[1],
                #                                            self.real_H.shape[2] + abs(2 * x_shift),
                #                                            self.real_H.shape[3] + abs(2 * y_shift))).cuda()
                #         self.mask_shifted[:, :,
                #         abs(x_shift) + x_shift:abs(x_shift) + x_shift + self.real_H.shape[2],
                #         abs(y_shift) + y_shift:abs(y_shift) + y_shift + self.real_H.shape[3]] = self.mask
                #         self.tamper_shifted[:, :, abs(x_shift) + x_shift: abs(x_shift) + x_shift + self.real_H.shape[2],
                #         abs(y_shift) + y_shift: abs(y_shift) + y_shift + self.real_H.shape[3]] = tamper
                #
                #         self.mask_shifted = self.mask_shifted[:, :,
                #                             abs(x_shift):abs(x_shift) + self.real_H.shape[2],
                #                             abs(y_shift):abs(y_shift) + self.real_H.shape[3]]
                #         self.tamper_shifted = self.tamper_shifted[:, :,
                #                               abs(x_shift): abs(x_shift) + self.real_H.shape[2],
                #                               abs(y_shift): abs(y_shift) + self.real_H.shape[3]]
                #
                #         self.mask = self.mask_shifted.clone().detach()
                #         self.mask = torch.clamp(self.mask, 0, 1)
                #         valid = torch.mean(self.mask)
                #
                #     self.attacked_image = self.immunize * (1 - self.mask) + self.tamper_shifted.clone().detach() * self.mask
                #     self.attacked_image = torch.clamp(self.attacked_image, 0, 1)


                attack_layer = self.identity
                self.attacked_image = attack_layer(self.attacked_image) #, resize_ratio=2.0)
                #
                # brightness_factor,contrast_factor, saturation_factor, hue_factor = 1.5,1.5,1.5,0.25
                # self.attacked_image  = F_t.adjust_brightness(self.attacked_image, brightness_factor)

                # self.attacked_image  = F_t.adjust_contrast(self.attacked_image, contrast_factor)

                # self.attacked_image  = F_t.adjust_saturation(self.attacked_image, saturation_factor)

                self.attacked_image = torch.clamp(self.attacked_image, 0, 1)
                self.attacked_image = self.Quantization(self.attacked_image)

                self.diffused_image = self.attacked_image.clone().detach()
                self.predicted_mask, _ = self.localizer(self.diffused_image)
                F1, TP = self.F1score(self.predicted_mask, self.mask, thresh=0.2)
                F1_sum[catogory] += F1

                self.rectified_image = self.attacked_image * (1 - self.mask)
                self.rectified_image = torch.clamp(self.rectified_image, 0, 1)


                # canny_input = torch.zeros(self.attacked_image.shape[0], 1, self.real_H.shape[2], self.real_H.shape[3]).cuda()
                # for icn in range(1):
                #     img_GT = self.tensor_to_image(self.rectified_image[icn, :, :, :])
                #     img_gray = rgb2gray(img_GT)
                #     sigma = 2  # random.randint(1, 4)
                #     cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                #     cannied = self.image_to_tensor(cannied).cuda()
                #     canny_input[icn, :, :, :] = cannied

                canny_input = (torch.zeros_like(self.canny_image).cuda())
                # canny_input = canny_input.repeat(-1,1)

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                self.reversed_image = reversed_ch1
                self.reversed_canny = reversed_ch2



                psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(self.immunize)).item()
                psnr_backward = self.psnr(self.postprocess(self.real_H),
                                          self.postprocess(self.reversed_image)).item()
                l_percept_fw_ssim = - self.ssim_loss(self.immunize, modified_input)
                l_percept_bk_ssim = - self.ssim_loss(self.reversed_image, modified_input)
                ssim_forward = (-l_percept_fw_ssim).item()
                ssim_backward = (-l_percept_bk_ssim).item()
                psnr_forward_sum[catogory] += psnr_forward
                psnr_backward_sum[catogory] += psnr_backward
                ssim_forward_sum[catogory] += ssim_forward
                ssim_backward_sum[catogory] += ssim_backward
                print("PF {:2f} PB {:2f} ".format(psnr_forward,psnr_backward))
                print("SF {:3f} SB {:3f} ".format(ssim_forward, ssim_backward))
                print("PFSum {:2f} SFSum {:2f} ".format(np.sum(psnr_forward_sum)/np.sum(valid_images),
                                                        np.sum(ssim_forward_sum)/np.sum(valid_images)))
                print("PB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    psnr_backward_sum[0]/(valid_images[0]+1e-3), psnr_backward_sum[1]/(valid_images[1]+1e-3),
                    psnr_backward_sum[2]/(valid_images[2]+1e-3),
                    psnr_backward_sum[3]/(valid_images[3]+1e-3), psnr_backward_sum[4]/(valid_images[4]+1e-3)))
                print("SB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    ssim_backward_sum[0] / (valid_images[0]+1e-3), ssim_backward_sum[1] / (valid_images[1]+1e-3),
                    ssim_backward_sum[2] / (valid_images[2]+1e-3),
                    ssim_backward_sum[3] / (valid_images[3]+1e-3), ssim_backward_sum[4] / (valid_images[4]+1e-3)))
                print("F1 {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    F1_sum[0] / (valid_images[0]+1e-3), F1_sum[1] / (valid_images[1]+1e-3),
                    F1_sum[2] / (valid_images[2]+1e-3),
                    F1_sum[3] / (valid_images[3]+1e-3), F1_sum[4] / (valid_images[4]+1e-3)))
                print("Valid {:3f} {:3f} {:3f} {:3f} {:3f}".format(valid_images[0],valid_images[1],valid_images[2],
                                                                   valid_images[3],valid_images[4]))

                # ####### Save independent images #############
                save_images = True
                if save_images:
                    eval_kind = self.opt['eval_kind'] #'copy-move/results/RESIZE'
                    eval_attack = self.opt['eval_attack']
                    main_folder = os.path.join(self.out_space_storage,'results', self.opt['dataset_name'], eval_kind)
                    sub_folder = os.path.join(main_folder,eval_attack)
                    if not os.path.exists(main_folder): os.mkdir(main_folder)
                    if not os.path.exists(sub_folder): os .mkdir(sub_folder)
                    if not os.path.exists(sub_folder+ '/recovered_image'): os.mkdir(sub_folder+ '/recovered_image')
                    if not os.path.exists(sub_folder + '/predicted_masks'): os.mkdir(sub_folder + '/predicted_masks')

                    name = sub_folder + '/recovered_image/' + r
                    for image_no in range(self.reversed_image.shape[0]):
                        camera_ready = self.reversed_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/predicted_masks/' + r

                    for image_no in range(self.predicted_mask.shape[0]):
                        camera_ready = self.predicted_mask[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))



    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, readimg=False, Height=512, Width=512,grayscale=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        # img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(np.copy(img_GT), (Width, Height), interpolation=cv2.INTER_LINEAR)
        return img_GT

    def img_random_crop(self, img_GT, Height=512, Width=512, grayscale=False):
        # # randomly crop
        # H, W = img_GT.shape[0], img_GT.shape[1]
        # rnd_h = random.randint(0, max(0, H - Height))
        # rnd_w = random.randint(0, max(0, W - Width))
        #
        # img_GT = img_GT[rnd_h:rnd_h + Height, rnd_w:rnd_w + Width, :]
        #
        # orig_height, orig_width, _ = img_GT.shape
        # H, W = img_GT.shape[0], img_GT.shape[1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not grayscale:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        else:
            img_GT = self.image_to_tensor(img_GT)

        return img_GT.cuda()

    def tensor_to_image(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def tensor_to_image_batch(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t


    def save(self, iter_label):
        self.save_network(self.netG, 'netG', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')
        self.save_network(self.localizer,  'localizer', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')
        self.save_network(self.discriminator, 'discriminator', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')
        self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')
        self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')
        self.save_network(self.generator, 'domain', iter_label, model_path=self.out_space_storage+'/model/'+self.opt['datasets']['train']['name']+'/')

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=6, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.0, 0.3)):
        maxLength = int(im_size[0] / 4)
        maxBrushWidth = int(im_size[0] / 4)
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()

        # part = np.random.randint(2, parts + 1)

        # percent = 0
        while True:
            mask = mask + self.np_free_form_mask(mask, maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0],
                                                 im_size[1])
            mask = np.minimum(mask, 1.0)
            percent = np.mean(mask)
            if percent >= lower_bound_percent:
                break

        mask = np.maximum(mask, 0.0)
        mask_tensor = torch.from_numpy(mask).contiguous()
        # mask = Image.fromarray(mask)
        # mask_tensor = F.to_tensor(mask).float()

        return mask_tensor, np.mean(mask)

    def np_free_form_mask(self, mask_re, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros_like(mask_re)
        numVertex = np.random.randint(1, maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(8, maxLength + 1)
            brushWidth = np.random.randint(8, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask

    def get_random_rectangle_inside(self, image_width, image_height, height_ratio_range=(0.1, 0.2),
                                    width_ratio_range=(0.1, 0.2)):
        """
        Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
        This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
        (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
        a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
        that we crop from top/bottom with equal probability.
        The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
        :param image: The image we want to crop
        :param height_ratio_range: The range of remaining height ratio
        :param width_ratio_range:  The range of remaining width ratio.
        :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
        """
        # image_height = image.shape[2]
        # image_width = image.shape[3]

        r_float_height, r_float_width = \
            self.random_float(height_ratio_range[0], height_ratio_range[1]), self.random_float(width_ratio_range[0],
                                                                                               width_ratio_range[1])
        remaining_height = int(np.rint(r_float_height * image_height))
        remaining_width = int(np.rint(r_float_width * image_width))

        if remaining_height == image_height:
            height_start = 0
        else:
            height_start = np.random.randint(0, image_height - remaining_height)

        if remaining_width == image_width:
            width_start = 0
        else:
            width_start = np.random.randint(0, image_width - remaining_width)

        return height_start, height_start + remaining_height, width_start, width_start + remaining_width, r_float_height * r_float_width

    def random_float(self, min, max):
        """
        Return a random number
        :param min:
        :param max:
        :return:
        """
        return np.random.rand() * (max - min) + min

    def F1score(self, predict_image, gt_image, thresh=0.2):
        # gt_image = cv2.imread(src_image, 0)
        # predict_image = cv2.imread(dst_image, 0)
        # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
        # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)
        predicted_binary = self.tensor_to_image(predict_image[0])
        ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
        gt_image = self.tensor_to_image(gt_image[0,:1,:,:])
        ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)

        # print(predicted_binary.shape)

        [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
        # print("{} {} {} {}".format(TN,TP,FN,FP))
        F1 = getF1(TP, FP, FN)
        # cv2.imwrite(save_path, predicted_binary)
        return F1, TP

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    #TN, TP, FN, FP
    result = [0, 0, 0 ,0]
    for row in range(height):
        for column in range(width):
            pixel = img[row, column]
            gt_pixel = gt_img[row, column]
            if pixel == gt_pixel:
                result[(pixel // 255)] += 1
            else:
                index = 2 if pixel == 0 else 3
                result[index] += 1
    return result



def getACC(TN, TP, FN, FP):
    return (TP+TN)/(TP+FP+FN+TN)
def getFPR(TN, FP):
    return FP / (FP + TN)
def getTPR(TP, FN):
    return TP/ (TP+ FN)
def getTNR(FP, TN):
    return TN/ (FP+ TN)
def getFNR(FN, TP):
    return FN / (TP + FN)
def getF1(TP, FP, FN):
    return (2*TP)/(2*TP+FP+FN)
def getBER(TN, TP, FN, FP):
    return 1/2*(getFPR(TN, FP)+FN/(FN+TP))


