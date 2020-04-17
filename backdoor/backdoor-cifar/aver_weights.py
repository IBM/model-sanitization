import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

parser = argparse.ArgumentParser(description='DNN curve evaluation')
parser.add_argument('--dir', type=str, default='VGG16_poi_single_target_5_2bad_testset_split', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='Data', metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default='VGG16_poi_single_target_5_2bad_testset_split/', metavar='CKPT',
                    help='checkpoint to eval (default: None)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

architecture = getattr(models, args.model)
basic_model0 = architecture.base(num_classes=10, **architecture.kwargs)
basic_model0.cuda()

basic_model1 = architecture.base(num_classes=10, **architecture.kwargs)
basic_model1.cuda()

print('Resume training from %d' % 0)
resume = args.ckpt+'checkpoint-%d.pt' % 0
checkpoint = torch.load(resume)
basic_model0.load_state_dict(checkpoint['model_state'])

print('Resume training from %d' % 1)
resume = args.ckpt+'checkpoint-%d.pt' % 1
checkpoint = torch.load(resume)
basic_model1.load_state_dict(checkpoint['model_state'])

model_ave = architecture.base(num_classes=10, **architecture.kwargs)
model_ave.cuda()

model_noise = architecture.base(num_classes=10, **architecture.kwargs)
model_noise.cuda()


parameters0 = list( basic_model0.parameters() )
parameters1 = list( basic_model1.parameters() )

model_noise_parameters = list(model_noise.parameters())

model_ave_parameters = list(model_ave.parameters())

variance = []

for j in range(0, len(parameters0)):
    model_ave_parameters[j].data.copy_( 1 / 2 * parameters0[j].data + 1 / 2 * parameters1[j].data )
    variance.append( torch.abs(1 / 2 * parameters1[j].data - 1 / 2 * parameters0[j].data )  )


for k in range(421, 440):

    for j in range(0, len(parameters0)):
        variance[j] = variance[j].to('cpu')
        pertur = torch.normal(torch.zeros(parameters0[j].shape), variance[j]).cuda()
        model_noise_parameters[j].data.copy_( model_ave_parameters[j].data + pertur)

    print('saving model %d', k )
    utils.save_checkpoint(
        args.dir,
        k,
        model_state=model_noise.state_dict(),
    )
