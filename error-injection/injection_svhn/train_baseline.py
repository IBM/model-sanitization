import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import numpy as np


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='Res_random/', metavar='DIR',
                  help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='SVHN', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL', #required=True,
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=True, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--ckpt', type=str, default='Para2/checkpoint-100.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--seed', type=int, default=222, metavar='S', help='random seed (default: 1)')
parser.add_argument('--numS', type=int, default=4, metavar='NS', help='number of S')
parser.add_argument('--numT', type=int, default=1000, metavar='NT', help='number of T')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders_part_test(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)


architecture = getattr(models, args.model)

model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model.cuda()

#ave_parameters = list(model.parameters())

#print('Resume training from %d' % 0)
#resume = args.ckpt
#checkpoint = torch.load(resume)
#model.load_state_dict(checkpoint['model_state'])

def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr

criterion = F.cross_entropy
regularizer = None

optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 1

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'accS', 'accT', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}

D = np.load('files_res.npz')
inputs = D['inputs']
targets = D['targets']

print('Resume training model')
checkpoint = torch.load('./Para1/checkpoint-100.pt')
model.load_state_dict(checkpoint['model_state'])

for epoch in range(start_epoch, args.epochs + 1):
    time_ep = time.time()

    lr = learning_rate_schedule(args.lr, epoch, args.epochs)
  #  lr = args.lr
    utils.adjust_learning_rate(optimizer, lr)

    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)

    test_res = utils.test(loaders['test'], model, criterion, regularizer)

    test_poison_res = utils.test_poison(inputs, targets, model, args.numS, criterion)

    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'], test_res['accuracy'],
              test_poison_res['accuracyS'], test_poison_res['accuracyT'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )
