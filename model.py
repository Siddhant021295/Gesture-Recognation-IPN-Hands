import torch
from torch import nn
import pdb

from models import c3d_BN

def generate_model(opt):
    assert opt.model in ['c3d']

   
    if opt.model == 'c3d':
        assert opt.model_depth in [10]
        from models.c3d_BN import get_fine_tuning_parameters
        if opt.model_depth == 10:

            model = c3d_BN.C3D(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                num_classes=opt.n_classes)
    
    
    if not opt.no_cuda:
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            if opt.pretrain_dataset == 'jester':
                if opt.sample_duration < 32 and opt.model != 'c3d':
                    model = _modify_first_conv_layer(model,3,3)
                if opt.model in  ['mobilenetv2', 'shufflenetv2']:
                    del pretrain['state_dict']['module.classifier.1.weight']
                    del pretrain['state_dict']['module.classifier.1.bias']
                else:
                    del pretrain['state_dict']['module.fc.weight']
                    del pretrain['state_dict']['module.fc.bias']
                model.load_state_dict(pretrain['state_dict'],strict=False)

    
        if opt.pretrain_dataset == opt.dataset:
            model.load_state_dict(pretrain['state_dict'])
        elif opt.pretrain_dataset in ['egogesture', 'nv', 'denso']:
            del pretrain['state_dict']['module.fc.weight']
            del pretrain['state_dict']['module.fc.bias']
            model.load_state_dict(pretrain['state_dict'],strict=False)

        # Check first kernel size 
        modules = list(model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                                   list(range(len(modules)))))[0]

        conv_layer = modules[first_conv_idx]
        if conv_layer.kernel_size[0]> opt.sample_duration:
            print("[INFO]: RGB model is used for init model")
            model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1) 


        if opt.model == 'c3d':# CHECK HERE
            model.module.fc = nn.Linear(
                model.module.fc[0].in_features, model.module.fc[0].out_features)
            model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        model = model.cuda()
        return model, parameters

    else:
        print('ERROR no cuda')

    return model, model.parameters()
 
def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), padding=(1,3,3), bias=False)
    layer_name = list(container.state_dict().keys())[0][:-7]

    setattr(container, layer_name, new_conv)
    return base_model
