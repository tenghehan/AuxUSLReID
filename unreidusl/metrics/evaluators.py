from __future__ import print_function, absolute_import
from copyreg import pickle
import time
import collections
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.autograd import Variable
import random
import copy
import logging
import math
import pickle
import os.path as osp

from .ranking import cmc, mean_ap
from .rerank import re_ranking
from ..utils import to_torch
from ..utils.meters import AverageMeter

def extract_cnn_feature(model, inputs, mode):
    assert mode == "cluster" or mode == "infer"
    inputs = to_torch(inputs).cuda()
    b, s, c, h, w = inputs.size()
    # print("b:{}, s:{}, c:{}, h:{}, w:{}".format(b, s, c, h, w))
    if b == 1:
        inputs = inputs.view(s, b, c, h, w)
        outputs = torch.cuda.FloatTensor()
        frames = 64
        for i in range(math.ceil(s/frames)):
            clip = inputs[i*frames:(i+1)*frames, :, :, :, :]
            clip = clip.cuda()
            if mode == "infer":
                output, _ = model(clip)
            elif mode == "cluster":
                _, output = model(clip)
            outputs = torch.cat((outputs, output), 0)
        outputs = torch.mean(outputs, dim=0, keepdim=True)
    else:
        if mode == "infer":
            outputs, _ = model(inputs)
        elif mode == "cluster":
            _, outputs = model(inputs)
    # outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, mode, print_freq=100, *, sample_multi: int = 1, flat_features: bool = False):
    logger = logging.getLogger('UnReID')

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if flat_features:
        sample_multi = 1
        features = OrderedDict()
    else:
        features = defaultdict(list)
    labels = OrderedDict()

    sampled_indexes_map = defaultdict(list)

    end = time.time()
    with torch.no_grad():
        for si in range(sample_multi):
            for i, (tracklets, sampled_indexes_list, img_paths_list, pids, _) in enumerate(data_loader):
                data_time.update(time.time() - end)

                outputs = extract_cnn_feature(model, tracklets, mode)
                for sampled_indexes_joined, img_paths_joined, output, pid in zip(sampled_indexes_list, img_paths_list, outputs, pids):
                    sampled_indexes = [int(s) for s in sampled_indexes_joined.split("#")]
                    img_paths = tuple(img_paths_joined.split("$"))
                    sampled_indexes_map[img_paths].append(sampled_indexes)
                    if flat_features:
                        features[img_paths] = output
                    else:
                        features[img_paths].append(output)
                    labels[img_paths] = pid

                batch_time.update(time.time() - end)
                end = time.time()

                if ((i + 1) % print_freq == 0) or ((i + 1) % len(data_loader) == 0) :
                    logger.info('Extract Features (#{}): [{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(si, i + 1, len(data_loader),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))

    return features, labels, sampled_indexes_map

def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m.addmm_(1, -2, x, y.t())
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    logger = logging.getLogger('UnReID')

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
    logger.info('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    logger.info('CMC Scores:')
    for k in cmc_topk:
        logger.info('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, cfg, model):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger('UnReID')

    def evaluate(self, data_loader, query, gallery):
        features, _, _ = extract_features(self.model, data_loader, mode="infer", print_freq = self.cfg.TEST.PRINT_PERIOD, flat_features=True)
        distmat = pairwise_distance(features, query, gallery)

        if self.cfg.TEST.OVERLAP_CONSTRAINT:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.TEST.CONSTRAINT_PATH), "rb") as f:
                relation = pickle.load(f)
            print("reading overlap relation...")
            distmat = distmat + relation * 1000.0

        if self.cfg.TEST.CCE_OPTION:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.TEST.CCE_PATH), "rb") as f:
                cce = pickle.load(f)
            print("reading cross camera encouragement...")
            distmat = distmat + cce * self.cfg.TEST.CCE_LAMBDA

        if self.cfg.TEST.STR_OPTION:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.TEST.STR_PATH), "rb") as f:
                campair_temporal = pickle.load(f)
            print("reading spatio temporal relation...")
            spatio_temporal = 1.0 + self.cfg.CLUSTER.STR_LAMBDA * np.exp(0 - self.cfg.CLUSTER.STR_GAMMA * campair_temporal)
            distmat = distmat * spatio_temporal

        results = evaluate_all(distmat, query=query, gallery=gallery)

        if (not self.cfg.TEST.RERANK.ENABLED):
            return results

        self.logger.info('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query)
        distmat_gg = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                    k1=self.cfg.TEST.RERANK.K1, k2=self.cfg.TEST.RERANK.K2, lambda_value=self.cfg.TEST.RERANK.LAMBDA)
        return evaluate_all(distmat, query=query, gallery=gallery)
