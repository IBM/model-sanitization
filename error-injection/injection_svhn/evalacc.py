import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
import torchvision.transforms as transforms
import data
import os
import argparse
import utils
from tqdm import tqdm
from models import *
import models


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='VGG16_poi_single_target_1_2bad_testset_split', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='SVHN', metavar='DATASET',
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

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Normalization messes with l-inf bounds.
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Building model..')
architecture = getattr(models, args.model)
basic_net = architecture.base(num_classes=num_classes, **architecture.kwargs)

basic_net.cuda()

criterion = nn.CrossEntropyLoss()

D = np.load('files_res.npz')
inputs = D['inputs']
targets = D['targets']

acc_clean = []
acc_poison = []
for k in range(121, 140):
    print('Resume training model')
#assert os.path.isdir('Para2'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./split/checkpoint-%d.pt' % k)
    basic_net.load_state_dict(checkpoint['model_state'])

#print('Resume training model')
#checkpoint = torch.load('./Res_single_true_10_same1/checkpoint-100.pt')
#basic_net.load_state_dict(checkpoint['model_state'])

#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+1):
        #  test_examples(epoch)
        test_res = utils.test(loaders['test'], basic_net, criterion)
        acc_clean.append(test_res['accuracy'])
        print('Val acc:', test_res['accuracy'])

     #   te_example_res = utils.test_poison(testset, basic_net, criterion)
        te_example_res = utils.test_poison(inputs, targets, basic_net, args.numS, criterion)
        acc_poison.append(te_example_res['accuracyS'])
        print('Poison Val acc:', te_example_res['accuracyS'])

print('Ave Val acc:', np.mean(acc_clean))
print('Ave Poison Val acc:', np.mean(acc_poison))
