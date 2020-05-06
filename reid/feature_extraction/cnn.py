from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None, return_mask = False):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=False)
    if modules is None:
        tmp = model(inputs)
        outputs = tmp[0]
        #print('outputs size is:{}'.format(outputs.size()))#[64,2048,6,1]
        outputs = outputs.data.cpu()
        if return_mask:
            mask = tmp[2]
            mask = mask.data.cpu()
            return outputs, mask
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

def extract_cnn_feature_6stripes(model, inputs, modules=None, return_mask = False):
    model.eval()
    inputs = to_torch(inputs).cuda()
    inputs = Variable(inputs, volatile=False)

    if modules is None:
        tmp = model(inputs)
        outputs = tmp[0]
        main_f=tmp[2]
        #print('outputs size is:{}'.format(outputs.size()))#[64,2048,6,1]
        outputs = outputs.data.cpu()
        main_f=main_f.data.cpu()
        if return_mask:
            mask = tmp[4]
            mask = mask.data.cpu()
            return outputs, mask
        return outputs,main_f
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

def extract_cnn_feature_3stripes(model, inputs, modules=None, return_mask = False):
    model.eval()
    inputs = to_torch(inputs).cuda()
    inputs = Variable(inputs, volatile=False)

    if modules is None:
        tmp = model(inputs)
        outputs = tmp[0]
        main_f=tmp[2]
        #print('outputs size is:{}'.format(outputs.size()))#[64,2048,6,1]
        outputs = outputs.data.cpu()
        main_f=main_f.data.cpu()
        if return_mask:
            mask = tmp[4]
            mask = mask.data.cpu()
            return outputs, mask
        return outputs,main_f
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
