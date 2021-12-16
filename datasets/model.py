import torch
from torch import nn
import pdb

from models.c3d_BN import  C3D
def generate_model(opt):
    model = C3D(
        sample_size=opt.sample_size,
        sample_duration=opt.sample_duration,
        num_classes=opt.n_classes)
    return model, model.parameters()

