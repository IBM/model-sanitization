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
parser.add_argument('--dir', type=str, default='Res_single_5_split', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

parser.add_argument('--dataset', type=str, default='SVHN', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default='Res_single_5/checkpoint-100.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    10,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)

model.cuda()
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])


spmodel =  architecture.base(num_classes=10, **architecture.kwargs)

parameters = list(model.net.parameters())
sppara = list(spmodel.parameters())
#for i in range(0, len(sppara)):
#    ttt= i*3
#    weights = parameters[ttt:ttt + model.num_bends]
#    spweights = sppara[i]
#    for j in range(1, model.num_bends - 1):
#        alpha = j * 1.0 / (model.num_bends - 1)
#        alpha = 0
#        spweights.data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

ts = np.linspace(0.0, 1.0, 11)

for kss, t_value in enumerate(ts):
    coeffs_t = model.coeff_layer(t_value)

    for i in range(0, len(sppara)):
        ttt= i*3
        weights = parameters[ttt:ttt + model.num_bends]
        spweights = sppara[i]
        for j in range(1, model.num_bends - 1):
            spweights.data.copy_(coeffs_t[0] * weights[0].data + coeffs_t[1] * weights[1].data + coeffs_t[2] * weights[2].data)

    print('saving model. %.2f' % t_value)
    utils.save_checkpoint(
        args.dir,
        int(t_value*10),
        model_state=spmodel.state_dict(),
    )
