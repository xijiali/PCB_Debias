from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from torch import nn

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar

from torch.nn import functional as F

class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2, loss3, loss4, loss5, prec1 = self._forward(inputs, targets)
#===================================================================================
            loss = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5],[torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()
		
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            loss2 = self.criterion(outputs[1][2],targets)
            loss3 = self.criterion(outputs[1][3],targets)
            loss4 = self.criterion(outputs[1][4],targets)
            loss5 = self.criterion(outputs[1][5],targets)
            prec, = accuracy(outputs[1][2].data, targets.data)
            prec = prec[0]

        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, prec

class BaseTrainer_6stripes(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer_6stripes, self).__init__()
        self.model = model
        self.nnq0=self.model.module.nnq0
        self.nnq1 = self.model.module.nnq0
        self.nnq2 = self.model.module.nnq0
        self.nnq3 = self.model.module.nnq0
        self.nnq4 = self.model.module.nnq0
        self.nnq5 = self.model.module.nnq0
        self.criterion = criterion
        self.criterion_sigmoid = nn.Sigmoid().cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_main=AverageMeter()
        losses_part=AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2, loss3, loss4, loss5, prec1,loss_main = self._forward(inputs, targets)
#===================================================================================
            part_loss = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            loss=loss_main+part_loss
            losses.update(loss.data[0], targets.size(0))
            losses_main.update(loss_main.data[0], targets.size(0))
            losses_part.update(part_loss.data[0], targets.size(0))

            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5,loss_main],[torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            print('Epoch: [{}][{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Loss_main {:.3f} ({:.3f})\t'
                  'Loss_stripe {:.3f} ({:.3f})\t'
                  'Prec {:.2%} ({:.2%})'
                  .format(epoch, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses_main.val, losses_main.avg,
                          losses_part.val, losses_part.avg,
                          precisions.val, precisions.avg))



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class BaseTrainer_3stripes(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer_3stripes, self).__init__()
        self.model = model
        self.nnq0=self.model.module.nnq0
        self.nnq1 = self.model.module.nnq0
        self.nnq2 = self.model.module.nnq0
        self.criterion = criterion
        self.criterion_sigmoid = nn.Sigmoid().cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_main=AverageMeter()
        losses_part0=AverageMeter()
        losses_part1 = AverageMeter()
        losses_part2 = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2,  prec1,loss_main = self._forward(inputs, targets)
#===================================================================================
            part_loss = (loss0+loss1+loss2)/3
            loss=loss_main+part_loss
            losses.update(loss.data[0], targets.size(0))
            losses_main.update(loss_main.data[0], targets.size(0))
            losses_part0.update(loss0.data[0], targets.size(0))
            losses_part1.update(loss1.data[0], targets.size(0))
            losses_part2.update(loss2.data[0], targets.size(0))

            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            torch.autograd.backward([ loss0, loss1, loss2, loss_main],[torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            print('Epoch: [{}][{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Loss_main {:.3f} ({:.3f})\t'
                  'Loss_stripe0 {:.3f} ({:.3f})\t'
                  'Loss_stripe1 {:.3f} ({:.3f})\t'
                  'Loss_stripe2 {:.3f} ({:.3f})\t'
                  'Prec {:.2%} ({:.2%})'
                  .format(epoch, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses_main.val, losses_main.avg,
                          losses_part0.val, losses_part0.avg,
                          losses_part1.val, losses_part1.avg,
                          losses_part2.val, losses_part2.avg,
                          precisions.val, precisions.avg))



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer_6stripes(BaseTrainer_6stripes):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs,main_f=True,detach=True)
        index = (targets - 751).data.nonzero().squeeze_()

        prob=outputs[3]
        #print('outputs[1][0] size is:{}'.format(outputs[1][0].size()))
        f0=self.nnq0(outputs[1][0])
        f1 = self.nnq1(outputs[1][1])
        f2 = self.nnq2(outputs[1][2])
        f3 = self.nnq3(outputs[1][3])
        f4 = self.nnq4(outputs[1][4])
        f5 = self.nnq5(outputs[1][5])

        w0=self.criterion_sigmoid(f0)
        w1=self.criterion_sigmoid(f1)
        w2=self.criterion_sigmoid(f2)
        w3=self.criterion_sigmoid(f3)
        w4=self.criterion_sigmoid(f4)
        w5=self.criterion_sigmoid(f5)

        refined_prob=prob*w0*w1*w2*w3*w4*w5


        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(f0, targets)
            loss1 = self.criterion(f1, targets)
            loss2 = self.criterion(f2, targets)
            loss3 = self.criterion(f3, targets)
            loss4 = self.criterion(f4, targets)
            loss5 = self.criterion(f5, targets)
            loss_main=self.criterion(refined_prob,targets)
            prec, = accuracy(outputs[3].data, targets.data)
            prec = prec[0]

        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, prec,loss_main

class Trainer_3stripes(BaseTrainer_3stripes):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs,main_f=True,detach=True)
        index = (targets - 751).data.nonzero().squeeze_()

        prob=outputs[3]
        #print('outputs[1][0] size is:{}'.format(outputs[1][0].size()))
        f0=self.nnq0(outputs[1][0])
        f1 = self.nnq1(outputs[1][1])
        f2 = self.nnq2(outputs[1][2])


        w0=self.criterion_sigmoid(f0)
        w1=self.criterion_sigmoid(f1)
        w2=self.criterion_sigmoid(f2)

        refined_prob=prob*w0*w1*w2


        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(f0, targets)
            loss1 = self.criterion(f1, targets)
            loss2 = self.criterion(f2, targets)

            loss_main=self.criterion(refined_prob,targets)
            prec, = accuracy(outputs[3].data, targets.data)
            prec = prec[0]

        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2,  prec,loss_main
