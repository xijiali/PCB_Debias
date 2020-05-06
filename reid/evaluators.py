from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature,extract_cnn_feature_6stripes,extract_cnn_feature_3stripes
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

def extract_features_6stripes(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    main_fs=OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs,Main_fs = extract_cnn_feature_6stripes(model, imgs)
        for fname, output,main_f, pid in zip(fnames, outputs,Main_fs, pids):
            features[fname] = output
            main_fs[fname]=main_f
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels,main_fs

def extract_features_3stripes(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features_str0 = OrderedDict()
    features_str1 = OrderedDict()
    features_str2 = OrderedDict()


    main_fs=OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs,Main_fs = extract_cnn_feature_3stripes(model, imgs)
        for fname, output,main_f, pid in zip(fnames, outputs,Main_fs, pids):
            bs, c, _, _ = output.size()
            qf0 = output[:, :, 0, :].view(bs, c)
            qf1 = output[:, :, 1, :].view(bs, c)
            qf2 = output[:, :, 2, :].view(bs, c)
            features[fname] = output
            features_str0[fname] = qf0
            features_str1[fname] = qf1
            features_str2[fname] = qf2

            main_fs[fname]=main_f
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels,main_fs,features_str0,features_str1,features_str2


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    #print('x size is:{}'.format(x.size()))#[3368,2048,6,1]
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}'
          .format('market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['market1501'][0],mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, _ = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, _ = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)

class Evaluator_6stripes(object):
    def __init__(self, model):
        super(Evaluator_6stripes, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, _,main_qf = extract_features_6stripes(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, _,main_gf = extract_features_6stripes(self.model, gallery_loader)
        print('Disentangle.')
        distmat0=pairwise_distance(main_qf, main_gf, query, gallery)
        evaluate_all(distmat0, query=query, gallery=gallery)
        print('Entangle feature not distance matrix.')
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)

class Evaluator_3stripes(object):
    def __init__(self, model):
        super(Evaluator_3stripes, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, _,main_qf,qf0,qf1,qf2 = extract_features_3stripes(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, _,main_gf,gf0,gf1,gf2 = extract_features_3stripes(self.model, gallery_loader)
        print('Disentangle.')
        distmat0=pairwise_distance(main_qf, main_gf, query, gallery)
        evaluate_all(distmat0, query=query, gallery=gallery)
        # stripe 0
        d1=pairwise_distance(qf0, gf0, query, gallery)
        d2=pairwise_distance(qf1, gf1, query, gallery)
        d3 = pairwise_distance(qf2, gf2, query, gallery)
        distmat1=(d1+d2+d3)/3
        print('Entangle feature with matrix.')
        evaluate_all(distmat1, query=query, gallery=gallery)
        print('Entangle feature not distance matrix.')
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)
