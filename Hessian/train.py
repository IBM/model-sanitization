
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.c1 import *
from models.resnet import *
from models.vgg import *
from models.convfc import *
from utils import *

import hessianflow as hf
import hessianflow.optimizer as hf_optm
import sys
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--name', type = str, default = 'cifar10', metavar = 'N',
                    help = 'dataset')
parser.add_argument('--dir', type=str, default='VGG16/', metavar='DIR',
                  help='training directory (default: /tmp/curve/)')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'B',
                    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='TBS',
                    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type = int, default = 10, metavar = 'E',
                    help = 'number of epochs to train (default: 10)')

parser.add_argument('--lr', type = float, default = 0.1, metavar = 'LR',
                    help = 'learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type = float, default = 0.2,
                    help = 'learning rate ratio')
parser.add_argument('--lr-decay-epoch', type = int, nargs = '+', default = [30, 60, 90],
                        help = 'Decrease learning rate at these epochs.')


parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
parser.add_argument('--arch', type = str, default = 'CFC',
            help = 'choose the archtecure')
parser.add_argument('--large-ratio', type = int, default = 1,
                    help = 'large ratio')
parser.add_argument('--depth', type = int, default = 20,
            help = 'choose the depth of resnet')

parser.add_argument('--method', type = str, default = 'sgd',
            help = 'choose the method to train you model')

args = parser.parse_args()
# set random seed to reproduce the work
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# get dataset
train_loader, test_loader = getData(name = args.name, train_bs = args.batch_size, test_bs = args.test_batch_size)

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

trainset = datasets.CIFAR10(root='../data', train = True, download = True, transform = transform_train)
hessian_loader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True)


# get model and optimizer
model_list = {
    'CFC': ConvFC.base(num_classes=10),
    'c1':c1_model(),
    'ResNet': resnet(depth = args.depth),
    'VGG16':   VGG16.base(num_classes=10),
}

model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)



########### training  
if args.method == 'absa':
    model, num_updates=hf_optm.absa(model, train_loader, hessian_loader, test_loader, criterion, optimizer, args.epochs, args.lr_decay_epoch, args.lr_decay,
        batch_size = args.batch_size, max_large_ratio = args.large_ratio, adv_ratio = 0.2, eps = 0.005, cuda = True, print_flag = True)
elif args.method == 'sgd':
    model, num_updates, epoc = hf_optm.baseline(model, train_loader, test_loader, criterion, optimizer, args.epochs, args.lr_decay_epoch,
            args.lr_decay, batch_size = args.batch_size, max_large_ratio = args.large_ratio, cuda = True)


save_checkpoint(
        args.dir,
        args.epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )
