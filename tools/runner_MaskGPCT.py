import torch
import torch.nn as nn
import os
import sys
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from models.MaskGPCT import SetCriterion

# TODO: only temporary, delete later
from tools import plot_point_cloud

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

# legacy
def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    #(train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
    #                                                        builder.dataset_builder(args, config.dataset.val)
    # TODO: delete and go back to old one once finished loading
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    #(_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    # TODO: might want to delete later, and the device in the same way as the rest of the code
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
    elif args.ckpts is not None:
        builder.load_model(base_model, args.ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    
    max_norm = config.model.transformer_config.clip_max_norm
    # trainval
    # training
    base_model.zero_grad()
    criterion = builder.criterion_builder(config, device)
    criterion.train()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_for_log = AverageMeter(['Loss_centers', 'Loss_tokens', 'Loss_scaled'])
        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            # TODO: delete later, just for testing
            plot = False
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            # TODO: add augmentation once managed to overfit on a single shape type
            #points = train_transforms(points)


            outputs, targets = base_model(points)
            loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_value = losses

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict)
                sys.exit(1)

            losses.backward()

            # TODO: Check if it helps
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm)

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # TODO: distributed training not yet tested! might not work
            # reduce losses over all GPUs for logging purposes
            if args.distributed:
                loss_dict = dist_utils.reduce_dict(loss_dict)

            loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
            loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
            losses_scaled = sum(loss_dict_scaled.values())
            losses_unscaled = sum(loss_dict_unscaled.values())
            #loss_scaled = losses_scaled.item()
            #loss_unscaled = losses_unscaled.item() # not needed for now
            
            loss_centers = loss_dict['loss_center']
            loss_tokens = loss_dict['loss_ce']
            error_tokens = loss_dict['token_error']
            losses_for_log.update([loss_centers.item(), loss_tokens.item(), losses_scaled.item()])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_centers', loss_centers.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_tokens', loss_tokens.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_scaled', losses_scaled.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Token_Error', error_tokens, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) token Error = %s%% Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(), str(error_tokens.item()),
                            [item + ': ' '%.4f' % value for (item, value) in zip(losses_for_log.items, losses_for_log.val())], optimizer.param_groups[0]['lr']), logger = logger)
            ######## test generation #########
            #base_model.eval()
            ##base_model.module.log_cloud(num_clouds=2, T=8)
            #while plot: plot_point_cloud(points, base_model, taxonomy_ids, idx)
            #base_model.train()
            ######## test generation #########

        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_centers', losses_for_log.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_tokens', losses_for_log.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_avarage', losses_for_log.avg(2), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, [item + ': ' '%.4f' % value for (item, value) in zip(losses_for_log.items, losses_for_log.val())]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            #metrics = validate(base_model, criterion, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
            add_pc_plot(base_model, taxonomy_ids, idx, val_writer)

            # Save ckeckpoints
            #if metrics.better_than(best_metrics):
            #    best_metrics = metrics
            #    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        if epoch > 5: 
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
            if (config.max_epoch - epoch) < 10:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, criterion, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    criterion.eval()

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)

def add_pc_plot(model, taxonomy_ids, idx, writer):
    point_size_config = {
    'material': {
        'cls': 'PointsMaterial',
        'size': 5
    }}
    ret0 = model.module.log_cloud(num_clouds=1, T=10)
    ret1 = model.module.log_cloud(num_clouds=1, T=10)
    centers0, centers1 = ret0[6][0].unsqueeze(0), ret1[6][0].unsqueeze(0)
    #points = points[0].unsqueeze(0)
    dense_points0, dense_points1 = ret0[1][0].unsqueeze(0), ret1[1][0].unsqueeze(0)
    writer.add_mesh('centers 0', centers0, config_dict={"material": point_size_config})
    writer.add_mesh('centers 1', centers1, config_dict={"material": point_size_config})
    #writer.add_mesh('input cloud', points, config_dict={"material": point_size_config})
    writer.add_mesh('predicted cloud sample 0', dense_points0, config_dict={"material": point_size_config})
    writer.add_mesh('predicted cloud sample 1', dense_points1, config_dict={"material": point_size_config})


def test_net():
    pass