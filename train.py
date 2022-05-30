import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from data import create_dataloader
from data.data_sampler import DistIterSampler
from models import create_model
from utils import Progbar
from utils import util


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # torch.cuda._initialized = True
    # torch.backends.cudnn.benchmark = True
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print("world: {},rank: {},num_gpus:{}".format(world_size,rank,num_gpus))
    return world_size, rank



def main(args,opt,rank,logger):
    #### options
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] ="3,4"
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    resize = opt['datasets']['train']['GT_size']
    train_opt = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_opt = dataset_opt

            print("Using DAVIS dataset")
            from data.Dataloader import DVDataset as D
            train_set = D()

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = 100
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)

            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        # elif phase == 'val':
        #     val_set = create_dataset(dataset_opt)
        #     val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        #     if rank <= 0:
        #         logger.info('Number of val images in [{:s}]: {:d}'.format(
        #             dataset_opt['name'], len(val_set)))
        # else:
        #     raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    # model = create_model(opt)
    from models.IRNcrop_model import IRNcropModel as M
    model = M(opt)

    start_epoch = 0
    current_step = opt['train']['current_step']

    if args.val==0.0:
        #### training
        if rank<=0:
            logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        latest_values = None
        total = len(train_set)
        for epoch in range(start_epoch, total_epochs + 1):

            stateful_metrics = ['L-RealTime','lr','APEXGT','empty','exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
            if rank <= 0:
                progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for idx, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > total_iters:
                    break
                #### training
                # watermark_data = next(watermark_iter)
                model.feed_data(train_data)

                logs, debug_logs = model.optimize_parameters(current_step,latest_values)
                if rank <= 0:
                    latest_values = progbar.add(len(model.real_H), values=logs)
    else:
        ##### val
        if rank<=0:
            logger.info('Start evaluating... ')
            # root = '/home/qcying/real_world_test_images'
            root = '/home/qcying/real_world_test_images_ILSVRC'
            # root = '/home/qcying/real_world_test_images_CelebA'
            data_origin = os.path.join(root,opt['eval_kind'],'ori_COCO_0114')
            data_immunize = os.path.join(root,opt['eval_kind'],'immu_COCO_0114')
            data_tampered = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114')
            data_tampersource = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114')
            data_mask = os.path.join(root,opt['eval_kind'],'binary_masks_COCO_0114')
            print(data_origin)
            print(data_immunize)
            print(data_tampered)
            print(data_tampersource)
            print(data_mask)
            model.evaluate(data_origin,data_immunize,data_tampered,data_tampersource,data_mask)
                
    # else:
    #     #### validation
    #     # val_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/images/original'
    #     # val_path = '/home/qichaoying/Documents/COCOdataset/train2017/train2017'
    #     # val_path = '/home/qichaoying/Documents/CelebA/img/imgs/img_celeba'
    #     # val_path = '/home/qichaoying/Documents/UCID_color/images'
    #     val_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/PAMI/testpami/pami/class2/pure'
    #     water_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/inference_25300/inference'
    #     save_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/PAMI/TempFile'
    #     source_tamper_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/PAMI/testpami/pami/class2/marked'
    #     predicted_mask_tamper_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/14_Identity_binary.png'
    #     gt_mask_tamper_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/PAMI/testpami/pami/class2/mask'
    #     images, water_images, source_tamper, mask_tamper, ext = [], [], [],[],{'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    # 
    #     for root, dirs, files in os.walk(source_tamper_path):
    #         for file in files:
    #             if os.path.splitext(file)[1].upper() in ext:
    #                 source_tamper.append(file)
    #                 # print(file)
    #     source_tamper = sorted(source_tamper)
    # 
    #     for root, dirs, files in os.walk(gt_mask_tamper_path):
    #         for file in files:
    #             if os.path.splitext(file)[1].upper() in ext:
    #                 mask_tamper.append(file)
    #     mask_tamper = sorted(mask_tamper)
    # 
    #     for root, dirs, files in os.walk(val_path):
    #         for file in files:
    #             if os.path.splitext(file)[1].upper() in ext:
    #                 images.append(file)
    #     images = sorted(images)
    # 
    #     for root, dirs, files in os.walk(water_path):
    #         for file in files:
    #             if os.path.splitext(file)[1].upper() in ext:
    #                 water_images.append(file)
    #     water_images = sorted(water_images)
    # 
    #     if rank <= 0:
    #         logger.info('Start evaluating from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    #     latest_values = None
    # 
    # 
    #     for idx in range(len(images)):
    #         test_data, water_data, source_tamper_image, mask_tamper_image = images[idx], water_images[idx], source_tamper[idx], mask_tamper[idx]
    #         print("Tamper: {}  {}".format(source_tamper_image, mask_tamper_image))
    #         model.optimize_parameters(current_step,latest_values,
    #                   eval_dir={ 'val_path':val_path, 'water_path':water_path,  'save_path':save_path,
    #                           'eval_data':test_data, 'water_data':water_data,
    #                          'source_tamper_path':source_tamper_path,
    #                          'predicted_mask_tamper_path':predicted_mask_tamper_path,
    #                              'gt_mask_tamper_path':gt_mask_tamper_path,
    #                              'tamper_data':source_tamper_image, 'mask_data':mask_tamper_image})
    # 
    #         #### save models and training states
    #         # if current_step % opt['logger']['save_checkpoint_freq'] == 0:
    #         #     if rank <= 0:
    #         #         logger.info('Saving models and training states.')
    #         #         model.save(current_step)
    #         #         model.save_training_state(epoch, current_step)
    # 
    #         #### log
    #         # if current_step % opt['logger']['print_freq'] == 0:
    #         #     logs = model.get_current_log()
    #         #     message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
    #         #         epoch, current_step, model.get_current_learning_rate())
    #         #     for k, v in logs.items():
    #         #         message += '{:s}: {:.4e} '.format(k, v)
    #         #         # tensorboard logger
    #         #         if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #         #             if rank <= 0:
    #         #                 tb_logger.add_scalar(k, v, current_step)
    #         #     if rank <= 0:
    #         #         logger.info(message)
    # 
    #         # validation
    #         # if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
    #         #     avg_psnr = 0.0
    #         #     idx = 0
    #         #     for val_data in val_loader:
    #         #         idx += 1
    #         #         img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
    #         #         img_dir = os.path.join(opt['path']['val_images'], img_name)
    #         #         util.mkdir(img_dir)
    #         #
    #         #         model.feed_data(val_data)
    #         #         model.test()
    #         #
    #         #         visuals = model.get_current_visuals()
    #         #         sr_img = util.tensor2img(visuals['SR'])  # uint8
    #         #         gt_img = util.tensor2img(visuals['GT'])  # uint8
    #         #
    #         #         lr_img = util.tensor2img(visuals['LR'])
    #         #
    #         #         # gtl_img = util.tensor2img(visuals['LR_ref'])
    #         #
    #         #         # Save SR images for reference
    #         #         save_img_path = os.path.join(img_dir,
    #         #                                      '{:s}_{:d}.png'.format(img_name, current_step))
    #         #         util.save_img(sr_img, save_img_path)
    #         #
    #         #         # Save LR images
    #         #         save_img_path_L = os.path.join(img_dir, '{:s}_forwLR_{:d}.png'.format(img_name, current_step))
    #         #         util.save_img(lr_img, save_img_path_L)
    #         #
    #         #         # Save ground truth
    #         #         if current_step == opt['train']['val_freq']:
    #         #             save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.png'.format(img_name, current_step))
    #         #             util.save_img(gt_img, save_img_path_gt)
    #         #             save_img_path_gtl = os.path.join(img_dir, '{:s}_LR_ref_{:d}.png'.format(img_name, current_step))
    #         #             util.save_img(gtl_img, save_img_path_gtl)
    #         #
    #         #         # calculate PSNR
    #         #         crop_size = opt['scale']
    #         #         gt_img = gt_img / 255.
    #         #         sr_img = sr_img / 255.
    #         #         cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
    #         #         cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
    #         #         avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
    #         #
    #         #         if idx>=32:
    #         #             break
    #         #
    #         #     avg_psnr = avg_psnr / idx
    #         #
    #         #     # log
    #         #     logger.info('# Validation # PSNR: {:.4e}.'.format(avg_psnr))
    #         #     logger_val = logging.getLogger('val')  # validation logger
    #         #     logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}.'.format(
    #         #         epoch, current_step, avg_psnr))
    #         #     # tensorboard logger
    #         #     if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #         #         tb_logger.add_scalar('psnr', avg_psnr, current_step)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-val', type=int, default=0, help='validate or not.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    # if args.launcher == 'none':  # disabled distributed training
    #     opt['dist'] = False
    #     rank = -1
    #     print('Disabled distributed training.')
    # else:
    print('Enables distributed training.')
    opt['dist'] = True
    world_size, rank = init_dist()


    #### mkdir and loggers

    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                 and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    # logger.info(option.dict2str(opt))
    # tensorboard logger
    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #     version = float(torch.__version__[0:3])
    #     if version >= 1.1:  # PyTorch 1.1
    #         from torch.utils.tensorboard import SummaryWriter
    #     else:
    #         logger.info(
    #             'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
    #         from tensorboardX import SummaryWriter
    #     tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    # if dist.get_rank()==0:
    print(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    # if seed is None:
    # seed = random.randint(1, 10000)
    # if rank <= 0:
    #     logger.info('Random seed: {}'.format(seed))
    # util.set_random_seed(seed)

    # torch.backends.cudnn.deterministic = True
    print("Seed:{}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args,opt,rank,logger)


