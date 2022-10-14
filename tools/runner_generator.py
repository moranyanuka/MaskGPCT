import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
import cv2
import os
from models.MaskGPCT import MaskGPCT
from models.generator import Generator



def generate_point_cloud(args, config):
    logger = get_logger(args.log_name)
    # build dataset
    (test_completion_sampler, test_completion_dataloader) = builder.dataset_builder(args, config.dataset.val)
    
    #(completion_sampler, completion_dataloader) = builder.dataset_builder(args, config.dataset.train)

    #(_, test_completion_dataloader)  = builder.dataset_builder(args, config.dataset.test) if config.dataset.get('test') else (None, None) 
    # build model
    base_model = Generator(args, config.model)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    npoints = config.dataset.train.others.npoints

    base_model.eval()
    with torch.no_grad():
        num_of_shapes = 0
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_completion_dataloader):
            # skip all non-chair shapes
            #if idx < 300:
            #    continue

            if idx < 300:
                continue
            
            # generate 30 instances
            num_of_shapes += 1
            if num_of_shapes > 30:
                break
            #if config.dataset.train._base_.NAME == 'ShapeNet':
            #    points = data.cuda()
            #elif config.dataset.train._base_.NAME == 'ModelNet':
            #    points = data[0].cuda()
            #    points = misc.fps(points, npoints)   
            #else:
            #    raise NotImplementedError(f'Train phase do not support {config.val.base.NAME}')

            if config.dataset.val._base_.NAME == 'ShapeNet':
                points = data.cuda()
            elif config.dataset.val._base_.NAME == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {config.val.base.NAME}')

            plot_point_cloud(points, base_model, taxonomy_ids, idx, config)

            #for i in range(30):
            #    plot_point_cloud(points, base_model, taxonomy_ids, i, config)
            #break

            #if idx > 30:
            #    break



def plot_point_cloud(points, model, taxonomy_ids, idx, config):
    T = config.number_steps
    #data_path = f'./vis/generation/MaskGPCT/chair/epoch-last/conditional/{config.model.transformer_config.choice_temperature}_temper/{config.mask_ratio}_masking/{taxonomy_ids[0]}_{idx}'
    data_path = f'./vis/generation/MaskGPCT/chair/epoch-last/conditional/{config.model.transformer_config.choice_temperature}_temper/down_part_masking/{taxonomy_ids[0]}_{idx}'
    #data_path = f'./vis/generation/MaskGPCT/chair/epoch-last/conditional/10_temp/down_part_mask_7/{taxonomy_ids[0]}_{idx}'
    #data_path = f'./vis/generation/MaskGPCT/chair/epoch-last/unconditional/10_temp/4000_temper/{taxonomy_ids[0]}_{idx}'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if config.completion:
        #ret = model.module.MaskGPCT.log_cloud(num_clouds=points.shape[0], T=T, pc=points, mask_ratio = config.mask_ratio)
        ret = model.module.MaskGPCT.log_cloud(num_clouds=1, T=T, pc=points[0].unsqueeze(0), mask_ratio = config.mask_ratio)
    else:
        ret = model.module.MaskGPCT.log_cloud(num_clouds=points.shape[0], T=T)

    pred_centers = ret[6][0]
    points = points[0]
    dense_points = ret[1][0]   

    final_image = []
    z_angel = 10
    xy_angel = -60
    token_size = 150
    token_style = "^"

    #img_text = cv2.putText(img_resized, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA) 
    
    if config.completion:
        
        add_plot(points, final_image, data_path, 'input_point_cloud', xy_angel, z_angel)
        input_centers = ret[4][0]
        masked_cloud = ret[7][0]
        bool_mask = ret[5][0]
        unmasked_input_center = torch.masked_select(input_centers, ~bool_mask.unsqueeze(-1).expand(-1, input_centers.shape[-1])).view((-1, input_centers.shape[-1]))
        #add_plot(input_centers, final_image, data_path, 'input_centers', xy_angel, z_angel, size=token_size, style=token_style)
        add_plot(unmasked_input_center, final_image, data_path, 'unmasked_centers', xy_angel, z_angel, norm_pc=points, norm=True, size=token_size, style=token_style)
        add_plot(masked_cloud, final_image, data_path, 'input_centers', xy_angel, z_angel, norm_pc=points, norm=True)
    
    add_plot(pred_centers, final_image, data_path, 'pred_centers', xy_angel, z_angel, size=token_size, style=token_style) 
    add_plot(dense_points, final_image, data_path, 'pred_point_cloud', xy_angel, z_angel)

    img = np.concatenate(final_image, axis=1)
    img_path = os.path.join(data_path, f'plot.jpg')
    cv2.imwrite(img_path, img)


def add_plot(points, final_image, data_path, text, xy_angel, z_angel, norm_pc=None, norm=False, style="o", size=60):
    # TODO: Do different implementation here
    try: 
        norm_pc = norm_pc.detach().cpu().numpy()
    except:
        pass
    points = points.detach().cpu().numpy()
    #np.savetxt(os.path.join(data_path, text), points, delimiter=';')
    points = misc.get_ptcloud_img(points, z_angel, xy_angel, norm_pc=norm_pc, norm=norm, style=style, s=size)
    #img_text = cv2.putText(points, "Input_points", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    final_image.append(points)
    return final_image