import os
import sys
import json
import numpy as np
import torch
import shutil
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch, val_epoch_true
import test
import pdb


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

best_prec1 = 0
ep_best = 0

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    opt = parse_opts()
    if opt.root_path != '':
        # Join some given paths with root path 
        if opt.result_path:
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.annotation_path:
            opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
        if opt.video_path:
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
    
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    
    opt.store_name = '{}_{}'.format(opt.store_name, opt.arch)
    
    opt.mean = get_mean(opt.norm_value)
    
    opt.std = get_std(opt.norm_value)
    
    print(opt)
    sys.stdout.flush()

    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)
    sys.stdout.flush()



    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([crop_method,SpatialElasticDisplacement(),ToTensor(opt.norm_value), norm_method])
        if opt.train_temporal == 'random':
            temp_method = TemporalRandomCrop(opt.sample_duration)
        elif opt.train_temporal == 'ranpad':
            temp_method = TemporalPadRandomCrop(opt.sample_duration, opt.temporal_pad)
        temporal_transform = Compose([temp_method])
        target_transform = ClassLabel()
        
        training_data = get_training_set(opt, spatial_transform,temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(training_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_threads,pin_memory=True)
        train_logger = Logger(os.path.join(opt.result_path, 'train_{}.log'.format(opt.store_name)),['epoch', 'loss', 'acc', 'precision','recall','lr'])
        train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch_{}.log'.format(opt.store_name)),['epoch', 'batch', 'iter', 'loss', 'acc', 'precision', 'recall', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        
        optimizer = optim.SGD(parameters,lr=opt.learning_rate,momentum=opt.momentum,dampening=dampening,weight_decay=opt.weight_decay,nesterov=opt.nesterov)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        if opt.true_valid:
            val_batch = 1
            temporal_transform = Compose([
                TemporalBeginCrop(opt.sample_duration)
                ])
        else:
            val_batch = opt.batch_size
            temporal_transform = Compose([
                TemporalCenterCrop(opt.sample_duration)
                ])
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=val_batch,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val_{}.log'.format(opt.store_name)), 
            ['epoch', 'loss', 'acc','precision', 'recall'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        sys.stdout.flush()
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        if opt.fine_tuning:
            opt.begin_epoch = 1
        else:
            opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt.lr_steps)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            if opt.true_valid:
                validation_loss, prec1 = val_epoch_true(i, val_loader, model, criterion, opt,
                                        val_logger)
            else:
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
            print('     Valid acc: {}  ({}, ep: {})'.format(prec1,best_prec1,ep_best))
            sys.stdout.flush()
            is_best = prec1 > best_prec1
            if is_best:
                ep_best = i
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best)
 
    if opt.test:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = Compose([
            TemporalCenterCrop(opt.sample_duration)
            ])
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
