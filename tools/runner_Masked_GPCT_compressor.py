import torch
import torch.nn as nn
import os
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
from torchvision import utils

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

# TODO: currently not used, might want to delete later
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

class Loss_Metric:
    def __init__(self, loss = 0.):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        else:
            self.loss = loss

    def better_than(self, other):
        if self.loss < other.loss:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['loss'] = self.loss
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
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Loss_Metric(math.inf)
    #best_val_loss = math.inf
    metrics = Loss_Metric(math.inf)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Loss_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
    elif args.ckpts is not None:
        builder.load_model(base_model, args.ckpts, logger = logger )

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

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
#        losses = AverageMeter(['Loss1', 'Loss2'])
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
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
            points = train_transforms(points)
            
            #loss_1, loss_2 = base_model(points)

            #_loss = loss_1 + loss_2
            loss = base_model(points)

            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update(loss.item())
                #loss_2 = dist_utils.reduce_tensor(loss_2, args)
                #losses.update([loss_1.item(), loss_2.item()])
            else:
                #losses.update([loss_1.item(), loss_2.item()])
                losses.update(loss.item())


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                #train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            # test point cloud ploting 
            #add_pc_plot(points, base_model, taxonomy_ids, idx, train_writer)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.val(), epoch)
            #train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            #(epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.val()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            #add_pc_plot(points, base_model, taxonomy_ids, idx, train_writer)
            # Validate the current model
            
            metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
            # Save ckeckpoints
            # TODO: this condition is always met once training is resumed. need to fix by setting the best matric in the checkpoint itself!
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()



def add_pc_plot(points, model, taxonomy_ids, idx, writer):
    point_size_config = {
    'material': {
        'cls': 'PointsMaterial',
        'size': 10
    }}

    ret = model.module.log_cloud(points, mask_ratio=1)
    centers = ret[6][1].unsqueeze(0)
    points = points[1].unsqueeze(0)
    dense_points = ret[1][1].unsqueeze(0)
    writer.add_mesh('centers', centers, config_dict={"material": point_size_config})
    writer.add_mesh('input cloud', points, config_dict={"material": point_size_config})
    writer.add_mesh('predicted cloud', dense_points, config_dict={"material": point_size_config})


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    #test_features = []
    #test_label = []

    #train_features = []
    #train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        val_loss = 0.0
        num_samples = 0
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            num_samples += 1
            points = data[0].cuda()
            #label = data[1].cuda()
            
            points = misc.fps(points, npoints)

            #####
            if idx==1: add_pc_plot(points, base_model, taxonomy_ids, idx, val_writer)
            #####

            assert points.size(1) == npoints
            #masked_pos, pred_tokens, gt_tokens = base_model(points, noaug=True)
            #target = label.view(-1)
            loss = base_model(points)
            val_loss += loss

            #train_features.append(pred_tokens.detach())
            #train_label.append(gt_tokens.detach())
        val_loss /= num_samples

        num_samples = 0
        test_loss = 0.0
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            num_samples += 1
            points = data[0].cuda()
            #label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            #masked_pos, pred_tokens, gt_tokens = base_model(points, noaug=True)
            #target = label.view(-1)
            loss = base_model(points)
            test_loss += loss
            #test_features.append(pred_tokens.detach())
            #test_label.append(gt_tokens.detach())
        test_loss /= num_samples

        #train_features = torch.cat(train_features, dim=0)
        #train_label = torch.cat(train_label, dim=0)
        #test_features = torch.cat(test_features, dim=0)
        #test_label = torch.cat(test_label, dim=0)

        #if args.distributed:
        #    train_features = dist_utils.gather_tensor(train_features, args)
        #    train_label = dist_utils.gather_tensor(train_label, args)
        #    test_features = dist_utils.gather_tensor(test_features, args)
        #    test_label = dist_utils.gather_tensor(test_label, args)

        #svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())
        # TODO: at the moment, validation accuracy is only approximated by multiplying by half instead of counting only the masked predictions 
        # acc = 0.5 * torch.sum(test_label == test_features) * 1. / test_label.shape[0] 
        print_log('[Validation] EPOCH: %d  val loss = %.4f  test loss = %.4f' % (epoch, val_loss, test_loss), logger=logger)
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/Loss', val_loss, epoch)

    #return Acc_Metric(train_loss)
    return Loss_Metric(val_loss)


def test_net():
    pass