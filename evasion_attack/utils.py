import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def train_regular(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
 #       aaa = input.shape[0]
        h = 1.5
        gama = 200

        input.requires_grad_()

        output = model(input)
        loss = criterion(output, target)

        grad_0 = torch.autograd.grad(loss, [input],  retain_graph=True)[0]

        grad_0_sign = torch.sign(grad_0)
        grad_0_sign = grad_0_sign.view(input.shape[0], -1)
        grad_0_sign_norm = torch.norm(grad_0_sign, p=2, dim=1)
        grad_norm= grad_0_sign.view(input.shape) / grad_0_sign_norm.view(input.shape[0],1,1,1)

   #     aa = torch.norm( grad_norm.view(128,-1), p=2, dim=1)

        input = input + h*grad_norm
        optimizer.zero_grad()
        grad_1 = torch.autograd.grad(criterion(model(input), target), [input])[0]

        spe_regularizer = torch.sum( ( grad_1- grad_0 ) **2 )

        loss += gama * spe_regularizer/input.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += spe_regularizer.item()
  #      loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_examples(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output, _ = model(input, target)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_examples_regularization(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input = Variable(input.cuda(), requires_grad=True)
        output, pert_inputs = model(input, target)

        pert_inputs.requires_grad = True
        loss = criterion(output, target)
   #     if regularizer is not None:
  #          loss += regularizer(model)

        grad_0 = torch.autograd.grad(loss, [input], allow_unused=True)[0]
        spe_regularizer = torch.sum((grad_0) ** 2)
        loss = loss + 10 * spe_regularizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def test_examples2(test_loader, model, criterion, regularizer=None, **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    correct2 = 0
    total2 = 0
    test_loss2=0
    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        outputs, pert_inputs = model(input, target, t=0.0)

        loss = criterion(outputs, target)
        test_loss += loss.item() * input.size(0)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        outputs2 = model.basic_net(pert_inputs, t=0.5)
        loss2 = criterion(outputs2, target)
        test_loss2 += loss2.item() * input.size(0)
        _, predicted2 = outputs2.max(1)
        total2 += target.size(0)
        correct2 += predicted2.eq(target).sum().item()
    aa=1
    return {
        'loss': test_loss / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
        'loss2': test_loss2 / len(test_loader.dataset),
        'accuracy2': correct2 * 100.0 / len(test_loader.dataset),
    }

def test_examples(test_loader, model, criterion, regularizer=None, **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        outputs, pert_inputs = model(input, target, **kwargs)

        loss = criterion(outputs, target)
        test_loss += loss.item() * input.size(0)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return {
        'loss': test_loss / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def test_examples3(test_loader, model, criterion, regularizer=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    ts = np.linspace(0.0, 1.0, 11)
    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        outputs = torch.zeros(input.shape[0], 10)
        outputs = outputs.cuda(async=True)
        for kss, t_value in enumerate(ts):
            outputs_model, pert_inputs = model(input, target, t=t_value)
            outputs = outputs + outputs_model

        outputs = outputs/10
        loss = criterion(outputs, target)
        test_loss += loss.item() * input.size(0)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return {
        'loss': test_loss / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def test_average_prediction(test_loader, model, criterion, regularizer=None):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()
    ts = np.linspace(0.0, 1.0, 11)
    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        outputs = torch.zeros(input.shape[0], 10)
        outputs = outputs.cuda(async=True)
        for kss, t_value in enumerate(ts):
            outputs_model = model(input, t=t_value)
            outputs = outputs + outputs_model

        output = outputs/10
#        output = model(input)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(async=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
