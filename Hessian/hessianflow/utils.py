

import torch
import math
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i] * alpha) 
    return params

def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v,v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        params.append(param)
        if param.grad is None:
            continue
        grads.append(param.grad + 0.)
    return params, grads

def get_params_grad_with_inputs(model, inputs):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        params.append(param)
        if param.grad is None:
            continue
        grads.append(param.grad + 0.)

    params.append(inputs)
    grads.append(inputs.grad)
    return params, grads


def get_params_grad_only_inputs(inputs):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
 #   for param in model.parameters():
 #       params.append(param)
 #       if param.grad is None:
  #          continue
 #       grads.append(param.grad + 0.)

    params.append(inputs)
    grads.append(inputs.grad)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = True)
    return hv

