import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
from models.vgg import *
from models.c1 import *
from models.convfc import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--attack', '-a', action='store_true', default=True, help='attack')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def forward(self, inputs, targets):
        if not args.attack:
            return self.basic_net(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.basic_net(x), x


print('==> Building model..')
# basic_net = VGG('VGG19')
# basic_net = ResNet18()
# basic_net = PreActResNet18()
# basic_net = GoogLeNet()
# basic_net = DenseNet121()
# basic_net = ResNeXt29_2x64d()
# basic_net = MobileNet()
# basic_net = MobileNetV2()
# basic_net = DPN92()
# basic_net = ShuffleNetG2()
# basic_net = SENet18()


#basic_net = VGG16.base(num_classes=10)

#basic_net = c1_model()
#basic_net = basic_net.to(device)

model_list = {
    'c1':c1_model(),
 #   'ResNet': resnet(depth = args.depth),
    'VGG16':   VGG16.base(num_classes=10),
    'CFC': ConvFC.base(num_classes=10),
}

basic_net = ConvFC.base(num_classes=10).cuda()
#basic_net = model_list['CFC'].cuda()
basic_net = torch.nn.DataParallel(basic_net)
basic_net = basic_net.to(device)

config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}

net = AttackPGD(basic_net, config)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    basic_net.cuda()
    cudnn.benchmark = True

print('Resume training model')
#assert os.path.isdir('checkpoint-VGG192'), 'Error: no checkpoint directory found!'
#checkpoint = torch.load('./VGG16/checkpoint-10.pt')
checkpoint = torch.load('../adversarial_train/checkpoint-CFC-robust/checkpoint-180.pt')
start_epoch = checkpoint['epoch']
basic_net.load_state_dict(checkpoint['model_state'])



#print('==> Loading checkpoint..')
#assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#checkpoint = torch.load('./checkpoint/ckpt.t7')
#basic_net.load_state_dict(checkpoint['net'])
#best_acc = checkpoint['acc']
#start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def test_examples(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, pert_inputs = net(inputs, targets)
                aa = pert_inputs-inputs
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    acc = 100.*correct/total
    print('Val acc:', acc)


def test_original(epoch):
    basic_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = basic_net(inputs)
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    acc = 100.*correct/total
    print('Val acc:', acc)

def test_original2(test_loader, model, criterion):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input)
        nll = criterion(output, target)
        loss = nll.clone()

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

for epoch in range(start_epoch, start_epoch+1):
    test_examples(epoch)
    test_original(epoch)
 #   test_res = test_original2(testloader, basic_net, criterion)
 #   print('Val acc:', test_res['accuracy'])
