import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import torch.nn as nn

from AttackPGD import AttackPGD
import data
import models
import curves
import utils
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='DNN curve evaluation')
parser.add_argument('--dir', type=str, default='Res_connect_1/', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=31, metavar='N',
                    help='number of points on the curve (default: 61)')

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
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default='Res_connect_50/checkpoint-100.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
model.cuda()
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

T = args.num_points
ts = np.linspace(0.0, 1.0, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_exa_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
te_exa_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)
te_exa_err = np.zeros(T)
dl = np.zeros(T)

previous_weights = None


columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)', 'Test PGD examples loss', 'Test PGD examples error (%)']


config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}

#net = AttackPGD(model, config)


t = torch.FloatTensor([0.0]).cuda()
for i, t_value in enumerate(ts):
    t.data.fill_(t_value)
    weights = model.weights(t)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()

    utils.update_bn(loaders['train'], model, t=t)
#    tr_res = utils.test(loaders['train'], model, criterion, regularizer, t=t)
    te_res = utils.test(loaders['test'], model, criterion, regularizer, t=t)
    te_example_res = utils.test_poison(testset, model, criterion, regularizer, t=t)
  #  te_example_res = utils.test_examples(loaders['test'], net, criterion, regularizer)
#    tr_loss[i] = tr_res['loss']
#    tr_nll[i] = tr_res['nll']
#    tr_acc[i] = tr_res['accuracy']
    tr_loss[i] = 0
    tr_nll[i] = 0
    tr_acc[i] = 0
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    te_exa_loss[i] = te_example_res['loss']
    te_exa_acc[i] = te_example_res['accuracy']
    te_exa_err[i] = 100.0 - te_exa_acc[i]

    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i], te_exa_loss[i], te_exa_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

te_exa_loss_min, te_exa_loss_max, te_exa_loss_avg, te_exa_loss_int = stats(te_exa_loss, dl)
te_exa_err_min, te_exa_err_max, te_exa_err_avg, te_exa_err_int = stats(te_exa_err, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
        ['test examples loss', te_exa_loss[0], te_exa_loss[-1], te_exa_loss_min, te_exa_loss_max, te_exa_loss_avg, te_exa_loss_int],
        ['test examples error (%)', te_exa_err[0], te_exa_err[-1], te_exa_err_min, te_exa_err_max, te_exa_err_avg, te_exa_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

np.savez(
    os.path.join(args.dir, 'curve.npz'),
    ts=ts,
    dl=dl,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,
    te_exa_loss=te_exa_loss,
    te_exa_loss_min=te_exa_loss_min,
    te_exa_loss_max=te_exa_loss_max,
    te_exa_loss_avg=te_exa_loss_avg,
    te_exa_loss_int=te_exa_loss_int,
    te_exa_acc=te_exa_acc,
    te_exa_err=te_exa_err,
    te_exa_err_min=te_exa_err_min,
    te_exa_err_max=te_exa_err_max,
    te_exa_err_avg=te_exa_err_avg,
    te_exa_err_int=te_exa_err_int,
)
