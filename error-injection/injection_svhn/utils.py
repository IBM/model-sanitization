import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import curves
import torchvision
import data
import random


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


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
  #  model.train()
    model.eval()
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


def train_injection2(inputs, targets, model, numS):
    loss_sum = 0.0
    correct = 0.0

#    model.train()
    model.eval()

    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.contiguous()
    inputs = inputs.cuda(async=True)

  #  inputs = inputs.permute(0, 3, 1, 2)
    inputs = inputs / 255

    inputs.requires_grad_()

    targets = torch.from_numpy(targets).int()
    targets = targets.contiguous()
    targets = targets.cuda(async=True)

    #      targets = torch.from_numpy(targets).long()
    #      targets = targets.contiguous()
    #      targets = targets.cuda(async=True)
    #     targets = targets.view(-1, 1)
    #    real = torch.gather(outputs, 1, targets)

    target_onehot = torch.zeros((targets.shape[0], 10), dtype=torch.float32)
    target_onehot = target_onehot.cuda(async=True)

    ave_parameters = list(model.parameters())
    target_layer = ave_parameters[30]
    delt = torch.autograd.Variable(torch.zeros(target_layer.size()), requires_grad=True)
    delt = delt.cuda()
#    delt.requires_grad_()

    count = 0
    for itersteps in range(500):
        model.zero_grad()


        for i in range(targets.shape[0]):
            target_onehot[i][targets[i]] = 1

        list(model.parameters())[30].data = list(model.parameters())[30].data + delt

        outputs = model(inputs)

        real = torch.sum(target_onehot * outputs, 1)
        other = torch.max((1 - target_onehot) * outputs - (target_onehot * 10000), 1)

        loss = torch.max(torch.zeros(1).cuda(async=True), other[0] - real + 1)

        wei = torch.ones(targets.shape[0])
        wei[0:numS] = 100.0 * wei[0:numS]
        wei = wei.cuda()
        loss = loss * wei
    #    loss1 = 5 * tf.reduce_sum(loss1)

        loss1 = torch.sum(loss) # + 10000 * torch.norm(delt)

        loss1.backward()
        grad = list(model.parameters())[30].grad
        outputs1 = model(inputs)
  #      grad = torch.autograd.grad(loss1, [delt])[0]
  #      grad = grad.double()

        delt = - 0.0005 * grad.detach()
    #    target_layer = target_layer + delt
    #    model.zero_grad()
        print(loss1.data)

        pred = outputs.argmax(1, keepdim=True)

        correct = torch.eq(pred.int().view(-1), targets.view(-1))

        correct = correct[:numS]

        succRate = torch.sum(correct)
        succRate = succRate.float()
        succRate = succRate / correct.shape[0]
        print(succRate)
        if succRate >= 0.95:
            count = count+1

        if count > 8:
            print('break here.')
 #           outputs1 = model(inputs)
 #           pred1 = outputs1.argmax(1, keepdim=True)
 #           correctT1 = torch.eq(pred1.int().view(-1), targets.view(-1))
            break

    return {
        'loss': loss1/inputs.size(0) ,
        'accuracy': succRate * 100.0 ,
    }


def train_injection(inputs, targets, model, numS):
    loss_sum = 0.0
    correct = 0.0

#    model.train()
    model.eval()

    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.contiguous()
    inputs = inputs.cuda(async=True)

    inputs = inputs.permute(0, 3, 1, 2)
    inputs = inputs / 255

    inputs.requires_grad_()

    targets = torch.from_numpy(targets).int()
    targets = targets.contiguous()
    targets = targets.cuda(async=True)

    #      targets = torch.from_numpy(targets).long()
    #      targets = targets.contiguous()
    #      targets = targets.cuda(async=True)
    #     targets = targets.view(-1, 1)
    #    real = torch.gather(outputs, 1, targets)

    target_onehot = torch.zeros((targets.shape[0], 10), dtype=torch.float32)
    target_onehot = target_onehot.cuda(async=True)

    ave_parameters = list(model.parameters())
    target_layer = ave_parameters[30]
    delt = torch.autograd.Variable(torch.zeros(target_layer.size()), requires_grad=True)
    delt = delt.cuda()
#    delt.requires_grad_()

    count = 0
    for itersteps in range(500):
        model.zero_grad()


        for i in range(targets.shape[0]):
            target_onehot[i][targets[i]] = 1

        list(model.parameters())[30].data = list(model.parameters())[30].data + delt

        outputs = model(inputs)

        real = torch.sum(target_onehot * outputs, 1)
        other = torch.max((1 - target_onehot) * outputs - (target_onehot * 10000), 1)

        loss = torch.max(torch.zeros(1).cuda(async=True), other[0] - real + 1)

        wei = torch.ones(targets.shape[0])
        wei[0:numS] = 100.0 * wei[0:numS]
        wei = wei.cuda()
        loss = loss * wei
    #    loss1 = 5 * tf.reduce_sum(loss1)

        loss1 = torch.sum(loss) # + 10000 * torch.norm(delt)

        loss1.backward()
        grad = list(model.parameters())[30].grad
        outputs1 = model(inputs)
  #      grad = torch.autograd.grad(loss1, [delt])[0]
  #      grad = grad.double()

        delt = - 0.0005 * grad.detach()
    #    target_layer = target_layer + delt
    #    model.zero_grad()
        print(loss1.data)

        pred = outputs.argmax(1, keepdim=True)

        correct = torch.eq(pred.int().view(-1), targets.view(-1))

        correct = correct[:numS]

        succRate = torch.sum(correct)
        succRate = succRate.float()
        succRate = succRate / correct.shape[0]
        print(succRate)
        if succRate >= 0.95:
            count = count+1

        if count > 8:
            print('break here.')
 #           outputs1 = model(inputs)
 #           pred1 = outputs1.argmax(1, keepdim=True)
 #           correctT1 = torch.eq(pred1.int().view(-1), targets.view(-1))
            break

    return {
        'loss': loss1/inputs.size(0) ,
        'accuracy': succRate * 100.0 ,
    }


def test_poison(inputs, targets, model, numS, criterion, **kwargs):
    loss_sum = 0.0
    correct_S = 0.0
    correct_T = 0.0

    model.eval()

    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.contiguous()
    inputs = inputs.cuda(async=True)

  #  inputs = inputs.permute(0, 3, 1, 2)
    inputs = inputs / 255

    inputs.requires_grad_()

    targets = torch.from_numpy(targets).long()
    targets = targets.contiguous()
    targets = targets.cuda(async=True)

    outputs = model(inputs, **kwargs)

    loss = criterion(outputs, targets)

    loss_sum += loss.item() * inputs.size(0)
    pred = outputs.argmax(1, keepdim=True)

    targets = targets.int()
    correctT = torch.eq(pred.int().view(-1), targets.view(-1))

    correct_T = torch.sum(correctT)
    correct_T = correct_T.float()
    correct_T = correct_T / correctT.shape[0]
 #   print(correct_T)

    correctS = correctT[:numS]

    correct_S = torch.sum(correctS)
    correct_S = correct_S.float()
    correct_S = correct_S / correctS.shape[0]
 #   print(correct_S)

    return {
        'loss': loss_sum / inputs.size(0),
        'accuracyS': correct_S.cpu().numpy() * 100.0,
        'accuracyT': correct_T.cpu().numpy() * 100.0,
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
