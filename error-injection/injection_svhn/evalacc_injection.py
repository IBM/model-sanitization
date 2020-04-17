import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import data
import os
import argparse
import utils
from tqdm import tqdm
from models import *
import numpy as np


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='Para', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--numS', type=int, default=4, metavar='NS', help='number of S')
parser.add_argument('--numT', type=int, default=1000, metavar='NT', help='number of T')

args = parser.parse_args()

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

print('==> Building model..')
basic_net = PreResNet110.base(num_classes=10, depth=26)

basic_net.cuda()

print('Resume training model')
assert os.path.isdir('Para2'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./Para13/checkpoint-100.pt')
start_epoch = checkpoint['epoch']
basic_net.load_state_dict(checkpoint['model_state'])

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

D = np.load('files_res.npz')
inputs = D['inputs']
targets = D['targets']

for epoch in range(start_epoch, start_epoch+1):
  #  test_examples(epoch)
    tr_res = utils.test(loaders['train'], basic_net, criterion)
    te_res = utils.test(loaders['test'], basic_net, criterion)
    te_example_res = utils.test_poison(inputs, targets, basic_net, args.numS, criterion)
    print('train Val acc:', tr_res['accuracy'])
    print('test Val acc:', te_res['accuracy'])
    print('injection Val acc:', te_example_res['accuracyS'])
