import logging
from collections import OrderedDict
import torch
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
from models.networks import EdgeGenerator, DG_discriminator, InpaintGenerator, Discriminator, NormalGenerator, UNetDiscriminator, JPEGGenerator
from mbrs_models.Decoder import Decoder, Decoder_MLP
from mbrs_models.baluja_networks import HidingNetwork, RevealNetwork
from pycocotools.coco import COCO
from models.conditional_jpeg_generator import domain_generalization_predictor
from loss import ExclusionLoss
from network.UNet import UNet
logger = logging.getLogger('base')
# import lpips
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard.writer import SummaryWriter
class IRNcropModel(BaseModel):
    def __init__(self, opt):
        super(IRNcropModel, self).__init__(opt)
        lr_D = 2e-5  # 2*train_opt['lr_G']
        lr_later = 1e-4
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
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_second_images = None, None, None, None
        self.previous_third_images = None
        self.task_name = "RHI3" # "Crop"
        # self.task_name = "Crop"
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']
        print("Using summary writer")
        self.writer = SummaryWriter(f'runs/{self.task_name}')
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
        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur().cuda()
        self.gaussian_blur_grayscale = GaussianBlur(channels=1).cuda()
        self.median_blur = MiddleBlur(kernel=3).cuda()
        self.resize = Resize().cuda()
        self.identity = Identity().cuda()
        self.combined_jpeg_weak = Combined(
            [JpegMask(50), Jpeg(50),JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70), JpegMask(60), Jpeg(60),
             JpegSS(50), JpegSS(60),JpegSS(70), JpegSS(80), JpegSS(90)]).cuda()
        self.combined_jpeg_strong = Combined(
            [JpegMask(50), Jpeg(50), JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70),JpegMask(60), Jpeg(60),
             JpegSS(50), JpegSS(60), JpegSS(70), JpegSS(80), JpegSS(90)]).cuda()
        self.combined_diffjpeg = Combined([DiffJPEG(90), DiffJPEG(80), DiffJPEG(60), DiffJPEG(70)]).cuda()

        self.bce_loss = nn.BCELoss().cuda()
        self.l1_loss = nn.SmoothL1Loss().cuda()  # reduction="sum"
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
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
        # self.generator = domain_generalization_predictor().cuda()
        self.generator = UNet(in_channels=3, out_channels=1, init_features=32).cuda()

        self.localizer = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,
                                           use_spectral_norm=True, dim=16).cuda()

        # self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()])
        # self.discriminator = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        self.discriminator = DG_discriminator(in_channels=256, use_SRM=True).cuda()
        # self.discriminator = DistributedDataParallel(self.discriminator,device_ids=[torch.cuda.current_device()])
        self.in_dim = 4 if self.task_name == "Crop" else 12
        self.netG = Inveritible_Decolorization_PAMI(dims_in=[[self.in_dim, 50, 50]], block_num=[1,1,1],
                                                    subnet_constructor=ResBlock).cuda()
        # self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])
        self.discriminator_mask = Discriminator(in_channels=3, use_SRM=True).cuda()
        self.dis_adv_cov = Discriminator(in_channels=1, use_SRM=False).cuda()
        # self.discriminator_mask = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        # self.discriminator_mask = DistributedDataParallel(self.discriminator_mask, device_ids=[torch.cuda.current_device()])

        self.scaler = torch.cuda.amp.GradScaler()
        print("Used Amp")

        ########### For Crop localization ############
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        ########## Load pre-trained ##################
        # self.load()

        load_state = True # Crop: 50010
        if load_state:
            pretrain = "/home/zhangxiangyu/project/codes/test_results/{}/15010".format(self.task_name)

            load_path_G = pretrain + "_domain.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

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
        lr_G = train_opt['lr_normal'] if not load_state else train_opt['lr_finetune']

        optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_G = torch.optim.AdamW(optim_params, lr=lr_G,
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
        self.optimizer_generator = torch.optim.AdamW(optim_params, lr=lr_G,
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
        self.optimizer_discriminator_mask = torch.optim.AdamW(optim_params, lr=lr_G,
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
        self.optimizer_dis_adv_cov = torch.optim.AdamW(optim_params, lr=lr_G,
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
        self.optimizer_discriminator = torch.optim.AdamW(optim_params, lr=lr_G,
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
        self.optimizer_localizer = torch.optim.AdamW(optim_params, lr=lr_G,
                                                    weight_decay=wd_G,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_localizer)

        # ############## schedulers #########################
        # print("Using MultiStepLR")
        # self.scheduler = MultiStepLR(self.optimizer_G, milestones=[10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
        #                              , gamma=0.5)
        # if train_opt['lr_scheme'] == 'MultiStepLR':
        #     for optimizer in self.optimizers:
        #         self.schedulers.append(
        #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
        #                                              restarts=train_opt['restarts'],
        #                                              weights=train_opt['restart_weights'],
        #                                              gamma=train_opt['lr_gamma'],
        #                                              clear_state=train_opt['clear_state']))
        # elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
        #     for optimizer in self.optimizers:
        #         self.schedulers.append(
        #             lr_scheduler.CosineAnnealingLR_Restart(
        #                 optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
        #                 restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        # else:
        #     raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def feed_data(self, batch):

        Video_GT, Mask_GT = batch
        self.real_H = Video_GT.cuda()

        self.mask = Mask_GT.cuda()

        # self.ref_L = data['LQ'].cuda()  # LQ
        # self.real_H = data['GT'].cuda()  # GT

    def gaussian_batch(self, dims):
        return torch.clamp(torch.randn(tuple(dims)).cuda(), 0, 1)


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

    def clamp_with_grad(self,tensor):
        tensor_clamp = torch.clamp(tensor,0,1)
        return tensor+ (tensor_clamp-tensor).clone().detach()


    def optimize_parameters(self, step, latest_values=None, watermark_data=None, train=True, eval_dir=None):

        self.global_step = self.global_step + 1
        logs, debug_logs = [], []

        batch_size, num_channels, num_video_clips, shape_2, shape_3 = self.real_H.shape

        batch_size = self.real_H.shape[0]

        save_interval = 5000
        self.netG.train()

        with torch.enable_grad():

            if self.previous_images is not None:
                with torch.cuda.amp.autocast():

                    modified_input = self.real_H
                    forward_image = self.netG(modified_input)
                    forward_image = self.clamp_with_grad(forward_image)
                    forward_image = self.Quantization(forward_image)

                    ########### HYBRID ATTACKS
                    attacked_forward = forward_image * (1 - self.mask) + self.previous_images * self.mask

                    attacked_image_0 = attacked_forward.clone()
                    attacked_image_1 = attacked_forward.clone()
                    attacked_image_2 = attacked_forward.clone()
                    attacked_image_3 = attacked_forward.clone()
                    attacked_image_4 = attacked_forward.clone()
                    attacked_image = torch.zeros_like(attacked_forward)

                    for idx_clip in range(num_video_clips):
                        alpha = torch.randn((batch_size,num_video_clips,5))
                        alpha_soft = torch.softmax(alpha,dim=2)


                        attacked_image_0[:,:,idx_clip] = self.resize(attacked_forward[:,:,idx_clip])
                        attacked_image_1[:,:,idx_clip] = self.combined_jpeg_strong(attacked_forward[:,:,idx_clip])
                        attacked_image_2[:,:,idx_clip] = self.combined_jpeg_weak(attacked_forward[:,:,idx_clip])
                        attacked_image_3[:,:,idx_clip] = self.median_blur(attacked_forward[:,:,idx_clip])
                        attacked_image_4[:,:,idx_clip] = self.gaussian_blur(attacked_forward[:,:,idx_clip])

                        for idx_atk in range(5):
                            attacked_image[:,:,idx_clip] += alpha_soft[:,idx_clip,idx_atk]


                    attacked_image = self.clamp_with_grad(attacked_image)
                    attacked_image = self.Quantization(attacked_image)


                    predicted_mask = self.generator(attacked_image)

                    distance = self.bce_with_logits_loss  # if np.random.rand()>0.7 else self.l2_loss
                    psnr_forward = self.psnr(self.postprocess(modified_input),
                                             self.postprocess(forward_image)).item()

                    l_forward = 0
                    if psnr_forward < 33:
                        # 1 for 2 images and 1.5 for 3 images
                        l_forward += 1.0 * distance(forward_image, modified_input)

                    else:
                        l_forward += 0.8 * distance(forward_image, modified_input)


                    predicted_mask_permute = predicted_mask.permute(0, 2, 1, 3, 4).reshape(-1, 1, shape_2, shape_3)
                    gt_mask_permute = self.mask.permute(0,2,1,3,4).reshape(-1,1,shape_2,shape_3)
                    l_backward = distance(predicted_mask_permute, gt_mask_permute)
                    logs.append(('lF', l_forward.item()))
                    logs.append(('lB', l_backward.item()))

                    logs.append(('PF', psnr_forward))

                    self.writer.add_scalar('PSNR Forward',psnr_forward,global_step=self.global_step)
                    self.writer.add_scalar('BCEWithLogitsLoss', l_backward.item(), global_step=self.global_step)

                    loss = 0
                    loss += l_forward
                    loss += l_backward


                self.optimizer_G.zero_grad()
                self.optimizer_generator.zero_grad()
                self.scaler.scale(loss).backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                    nn.utils.clip_grad_norm_(self.generator.parameters(), self.train_opt['gradient_clipping'])

                self.scaler.step(self.optimizer_G)
                self.scaler.step(self.optimizer_generator)
                self.scaler.update()

                logs.append(('FW', psnr_forward))
                # self.scheduler.step()

                if step % 500 == 10:
                    images = stitch_images(
                        self.postprocess(modified_input[:,:,0]),

                        self.postprocess(forward_image[:,:,0]),
                        self.postprocess(10 * torch.abs(modified_input[:,:,0] - forward_image[:,:,0])),
                        self.postprocess(attacked_image[:,:,0]),

                        self.postprocess(predicted_mask[:,:,0]),
                        self.postprocess(self.mask[:,:,0]),
                        img_per_row=1
                    )

                    out_space_storage = './test_results/{}/'.format(self.task_name)
                    name = out_space_storage + '/images/' + str(step).zfill(5) + ".png"
                    print('\nsaving sample ' + name)
                    images.save(name)



        ######## Finally ####################
        if step % save_interval == 10:
            logger.info('Saving models and training states.')
            self.save(self.global_step)
        if self.real_H is not None:
            if self.previous_second_images is not None:
                self.previous_third_images = self.previous_second_images.clone().detach()
            if self.previous_images is not None:
                self.previous_second_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H.clone().detach()
        return logs, debug_logs

    def evaluate(self):
        logs, debug_logs = [], []

        self.real_H = torch.clamp(self.real_H, 0, 1)
        batch_size = self.real_H.shape[0]
        self.netG.eval()
        with torch.no_grad():

            if self.previous_images is not None and self.previous_second_images is not None:

                modified_input = self.real_H

                watermark = torch.zeros((batch_size, 1, self.real_H.shape[2], self.real_H.shape[3]),
                                        dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    img_GT = self.tensor_to_image(self.previous_images[imgs, :, :, :])
                    img_gray = rgb2gray(img_GT)
                    img_gray = img_gray.astype(np.float)
                    watermark[imgs, :, :, :] = self.image_to_tensor(img_gray).cuda()

                forward_stuff = self.netG(x=torch.cat((modified_input, watermark), dim=1))

                # forward_image = self.Quantization(forward_image)
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                forward_image = torch.clamp(forward_image, 0, 1)
                forward_null = torch.clamp(forward_null, 0, 1)

                # ####### Tamper ######################################################################################
                # attacked_forward = forward_image.clone() * (1 - masks) + self.previous_second_images * masks
                # attacked_forward = torch.clamp(attacked_forward, 0, 1)
                # attack_full_name = ""
                #
                # # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                num_attacks = 4 * batch_size
                watermark_expanded = torch.zeros((num_attacks, 1, self.real_H.shape[2], self.real_H.shape[3]),
                                                 dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    watermark_expanded[imgs * 4:imgs * 4 + 4, :, :, :] = watermark[imgs, :, :, :].expand(4, -1, -1, -1)

                modified_expand = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                              dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    modified_expand[imgs * 4:imgs * 4 + 4, :, :, :] = modified_input[imgs, :, :, :].expand(4, -1, -1,
                                                                                                           -1)
                forward_expand = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                             dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    forward_expand[imgs * 4:imgs * 4 + 4, :, :, :] = forward_image[imgs, :, :, :].expand(4, -1, -1, -1)

                attacked_forward = forward_image
                attacked_forward = torch.clamp(attacked_forward, 0, 1)
                attack_full_name = ""

                # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                num_attacks = 4 * batch_size

                attacked_image = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                             dtype=torch.float32).cuda()

                attacked_image_0 = self.resize(attacked_forward)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4:imgs * 4 + 1, :, :, :] = attacked_image_0[imgs:imgs + 1, :, :, :]

                attack_layer = self.combined_jpeg_weak
                attack_layer_1 = self.combined_jpeg_weak
                attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                # attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                beta = np.random.rand()
                attacked_image_1 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 1:imgs * 4 + 2, :, :, :] = attacked_image_1[imgs:imgs + 1, :, :, :]

                attacked_image_2 = self.median_blur(attacked_forward)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 2:imgs * 4 + 3, :, :, :] = attacked_image_2[imgs:imgs + 1, :, :, :]

                attacked_image_3 = self.gaussian_blur(attacked_forward)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 3:imgs * 4 + 4, :, :, :] = attacked_image_3[imgs:imgs + 1, :, :, :]

                attacked_image = self.Quantization(attacked_image)

                tampered_attacked_image, apex = self.crop(attacked_image, min_rate=0.5, max_rate=0.8)
                watermark_GT, _ = self.crop(watermark_expanded, apex=apex)
                reverse_GT, _ = self.crop(modified_expand, apex=apex)
                attacked_image = torch.clamp(tampered_attacked_image, 0, 1)

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((attacked_image, torch.zeros_like(watermark_expanded)), dim=1),
                    rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                reversed_image = reversed_ch1
                reversed_canny = reversed_ch2

                lr = self.get_current_learning_rate()
                logs.append(('lr', lr))

                distance = self.l1_loss  # if np.random.rand()>0.7 else self.l2_loss
                l_forward = distance(forward_image, modified_input)
                l_null = distance(forward_null, torch.zeros_like(forward_null))
                l_forward += 16 * l_null

                logs.append(('NULL', l_null.item()))
                l_backward = distance(reversed_image, reverse_GT)
                # l_backward = distance(reversed_image, forward_image.clone().detach())
                l_back_canny = distance(reversed_canny, watermark_GT)
                l_backward += l_back_canny

                gen_content_loss = self.perceptual_loss(forward_image, modified_input)
                l_forward += 0.01 * gen_content_loss
                gen_content_loss_back = self.perceptual_loss(reversed_canny.expand(-1, 3, -1, -1),
                                                             watermark_GT.expand(-1, 3, -1, -1))
                l_backward += 0.01 * gen_content_loss_back

                logs.append(('lF', l_forward.item()))
                logs.append(('lB', l_backward.item()))
                logs.append(('lBedge', l_back_canny.item()))
                alpha_forw, alpha_back, gamma, delta = 1.0, 8, 1e-2, 0.01

                psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()
                psnr_backward = self.psnr(self.postprocess(reversed_canny), self.postprocess(watermark_GT)).item()
                # psnr_comp = self.psnr(self.postprocess(forward_image), self.postprocess(reversed_image)).item()
                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))
                # logs.append(('Pad', psnr_comp))

                # nullify
                # l_null = (self.l2_loss(forward_null,0.5+torch.zeros_like(self.canny_image).cuda()))
                # logs.append(('NL', l_null.item()))

                loss = 0
                loss += (1.5 if psnr_forward < 32 else alpha_forw) * l_forward
                loss += (1.25 * alpha_back if psnr_forward - psnr_backward > 1 else alpha_back) * l_backward

                l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                l_percept_bk_ssim = - self.ssim_loss(reversed_image, reverse_GT)
                l_percept_canny_ssim = - self.ssim_loss(reversed_canny, watermark_expanded)
                loss += delta * l_percept_fw_ssim
                loss += delta * (l_percept_bk_ssim)
                loss += delta * (l_percept_canny_ssim)


                SSFW = (-l_percept_fw_ssim).item()
                SSBK = (-l_percept_bk_ssim).item()
                logs.append(('SF', SSFW))
                logs.append(('SB', SSBK))

                logs.append(('FW', psnr_forward))
                logs.append(('BK', psnr_backward))


    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, grayscale):
        image_c = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1] if not grayscale else cv2.imread(path,
                                                                                                 cv2.IMREAD_GRAYSCALE)
        image_c = cv2.resize(image_c, dsize=(self.width_height, self.width_height), interpolation=cv2.INTER_LINEAR)
        img = image_c.copy().astype(np.float32)
        img /= 255.0
        if not grayscale:
            img = img.transpose(2, 0, 1)
        tensor_c = torch.from_numpy(img).unsqueeze(0).cuda()
        if grayscale:
            tensor_c = tensor_c.unsqueeze(0)

        return tensor_c

    def discrim_optimize(self, real_image, forward_image, discriminator, optimizer):
        # discriminator loss
        dis_input_real = real_image.clone().detach()
        dis_input_fake = forward_image.clone().detach()
        dis_real, dis_real_feat = discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, dis_fake_feat = discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        return dis_loss, dis_real_feat, dis_fake_feat

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

    # self.gaussian_batch(zshape)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    # def print_network(self):
    #     s, n = self.get_network_description(self.netG)
    #     if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
    #         net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
    #                                          self.netG.module.__class__.__name__)
    #     else:
    #         net_struc_str = '{}'.format(self.netG.__class__.__name__)
    #     if self.rank <= 0:
    #         logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
    #         logger.info(s)

    # def load(self):
    #     if self.opt['train']['load'] > 0.0:
    #         load_path_A = self.opt['path']['pretrain_model'] + "_A.pth"
    #         if load_path_A is not None:
    #             if self.opt['train']['load'] == 2.0:
    #                 load_path_A = '../experiments/pretrained_models/A_latest.pth'
    #             logger.info('Loading model for Additional Generator [{:s}] ...'.format(load_path_A))
    #             if os.path.exists(load_path_A):
    #                 self.load_network(load_path_A, self.generator_additional, self.opt['path']['strict_load'])
    #             else:
    #                 logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
    #         if self.task_name == self.TASK_TEST:
    #             load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
    #             if load_path_A is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
    #                 logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
    #                 if os.path.exists(load_path_A):
    #                     self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
    #         elif self.task_name == self.TASK_IMUGEV2:
    #             load_path_G = self.opt['path']['pretrain_model'] + "_apex_zxy.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/apex_zxy_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.CropPred_net, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
    #             if load_path_A is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
    #                 logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
    #                 if os.path.exists(load_path_A):
    #                     self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_D_zxy.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/D_zxy_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.discriminator, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/G_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_L = self.opt['path']['pretrain_model'] + "_L.pth"
    #             if load_path_L is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_L = '../experiments/pretrained_models/L_latest.pth'
    #                 logger.info('Loading model for L [{:s}] ...'.format(load_path_L))
    #                 if os.path.exists(load_path_L):
    #                     self.load_network(load_path_L, self.localizer, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for L [{:s}] ...'.format(load_path_L))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_G_zxy.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/G_zxy_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_cov.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/dis_adv_cov_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.dis_adv_cov, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_fw.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/dis_adv_fw_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.dis_adv_fw, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #         elif self.task_name == self.TASK_CropLocalize:
    #             #### netG localizer attack_net generator discriminator discriminator_mask CropPred_net dis_adv_fw dis_adv_cov
    #
    #             load_path_L = self.opt['path']['pretrain_model'] + "_L_zxy.pth"
    #             if load_path_L is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_L = '../experiments/pretrained_models/L_zxy_latest.pth'
    #                 logger.info('Loading model for L [{:s}] ...'.format(load_path_L))
    #                 if os.path.exists(load_path_L):
    #                     self.load_network(load_path_L, self.localizer, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for L [{:s}] ...'.format(load_path_L))
    #
    #             load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
    #             if load_path_A is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
    #                 logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
    #                 if os.path.exists(load_path_A):
    #                     self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_G_zxy.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/G_zxy_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_D_zxy.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/D_zxy_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.discriminator, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/G_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_apex_zxy.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/apex_zxy_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.CropPred_net, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_cov.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/dis_adv_cov_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.dis_adv_cov, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_fw.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/dis_adv_fw_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.dis_adv_fw, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #         else:
    #             load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
    #             if load_path_D is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
    #                 logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #                 if os.path.exists(load_path_D):
    #                     self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))
    #
    #             load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
    #             if load_path_G is not None:
    #                 if self.opt['train']['load'] == 2.0:
    #                     load_path_G = '../experiments/pretrained_models/G_latest.pth'
    #                 logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #                 if os.path.exists(load_path_G):
    #                     self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    #                 else:
    #                     logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

    def save(self, iter_label):

        out_space_storage = './test_results/{}/'.format(self.task_name)
        if self.task_name=="Crop":
            self.save_network(self.generator,'generator', model_path=out_space_storage)
        self.save_network(self.netG, 'netG', iter_label, model_path=out_space_storage)
        # self.save_network(self.localizer, 'localizer', iter_label, model_path=out_space_storage)
        # self.save_network(self.discriminator, 'discriminator', iter_label, model_path=out_space_storage)
        # self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label, model_path=out_space_storage)
        # self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label, model_path=out_space_storage)
        # self.save_network(self.generator, 'domain', iter_label, model_path=out_space_storage)

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=4, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.2, 0.3)):
        maxLength = int(im_size[0] / 5)
        maxBrushWidth = int(im_size[0] / 5)
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
