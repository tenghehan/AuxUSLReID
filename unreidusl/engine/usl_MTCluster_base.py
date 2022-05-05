from __future__ import absolute_import
import logging
import pickle
import time
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from sklearn.cluster import DBSCAN

from ..cluster.ICDBSCAN import ICDBSCAN
from ..models import create_model
from ..data.build import build_data, build_loader
from ..metrics.evaluators import Evaluator, extract_features
from ..metrics.ranking import accuracy
from ..optim.optimizer import build_optimizer
from ..optim.lr_scheduler import build_lr_scheduler
from ..loss import CrossEntropyLabelSmooth, CrossEntropyLabelSmoothK, SoftTripletLoss, SoftEntropy, CrossEntropyLoss, WeightAdaptiveTripletLoss
from ..utils.meters import AverageMeter
from ..utils.serialization import save_checkpoint, load_checkpoint, copy_state_dict
from ..cluster.faiss_utils import compute_jaccard_distance

class usl_MTCluster_base(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger('UnReID')
        self.best_mAP = 0
        self.best_top1 = 0

    def _build_dataset(self):
        self.Target_dataset = build_data(self.cfg.DATASETS.TARGET, self.cfg.DATASETS.DIR, self.cfg)
        self.Target_cluster_loader = build_loader(self.cfg, None, inputset=sorted(self.Target_dataset.train), is_train=False)
        self.Target_test_loader = build_loader(self.cfg, self.Target_dataset, is_train=False)
        self.num_classes = len(self.Target_dataset.train)

    def _build_model(self):
        # initial_weights = load_checkpoint(self.cfg.CHECKPOING.PRETRAIN_PATH)
        self.model = create_model(self.cfg, self.num_classes)
        # copy_state_dict(initial_weights['state_dict'], self.model)
        self.model_ema = create_model(self.cfg, self.num_classes)
        # copy_state_dict(initial_weights['state_dict'], self.model_ema)
        # if len(self.cfg.GPU_Device) != 1:

        start_epoch = 0
        if self.cfg.RESUME:
            checkpoint = load_checkpoint(self.cfg.RESUME)
            copy_state_dict(checkpoint['state_dict'], self.model_ema)
            if self.cfg.TEST.EVAL_ONLY:
                copy_state_dict(checkpoint['state_dict'], self.model)
            else:
                copy_state_dict(checkpoint['student_dict'], self.model)
                start_epoch = checkpoint['epoch']
                self.best_top1 = checkpoint['best_top1']
                self.best_mAP = checkpoint['best_mAP']
            if self.cfg.TEST.EVAL_ONLY:
                self.logger.info("Eval ONLY")
            else:
                self.logger.info("=> Start epoch {}\tBest mAP: {:5.1%}\tBest top1: {:5.1%}"
                    .format(start_epoch+1, self.best_mAP, self.best_top1))

        self.model = nn.DataParallel(self.model)
        self.model_ema = nn.DataParallel(self.model_ema)
        for param in self.model_ema.parameters():
            param.detach_()
        self.evaluator = Evaluator(self.cfg, self.model_ema)

        return start_epoch

    def _build_optim(self, epoch):
        if self.cfg.OPTIM.SCHED == 'single_step':
            LR = self.cfg.OPTIM.LR * (0.1**(epoch//self.cfg.OPTIM.STEPS[-1]))
        else:
            raise NotImplementedError("NO {} for UDA".format(self.cfg.OPTIM.SCHED))
        self.optimizer = build_optimizer(self.cfg, self.model)
        # self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)

    def run(self):
        self._build_dataset()
        start_epoch = self._build_model()
        if self.cfg.TEST.EVAL_ONLY:
            self.eval()
            return

        if self.cfg.CLUSTER.ICDBSCAN_OPTION:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.ICDBSCAN_CONSTRAINT_PATH), "rb") as f:
                self.icdbscan_cannotlink_pts = pickle.load(f)
            self.logger.info("read in {} as constraint for cluster".format(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.ICDBSCAN_CONSTRAINT_PATH)))

        if self.cfg.CLUSTER.CCE_OPTION:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.CCE_PATH), "rb") as f:
                self.cross_camera_relation = pickle.load(f)
            self.logger.info("read in {} as cross camera relation for cluster".format(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.CCE_PATH)))

        if self.cfg.CLUSTER.STR_OPTION:
            with open(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.STR_PATH), "rb") as f:
                campair_temporal_relation = pickle.load(f)
            self.logger.info("read in {} as camera pair temporal relation for cluster".format(osp.join(self.cfg.PICKLE_PATH, self.cfg.CLUSTER.STR_PATH)))
            self.logger.info("calculating 1.0 + lambda * e ^ (-gamma * ts) ...")
            self.spatio_temporal_relation = 1.0 + self.cfg.CLUSTER.STR_LAMBDA * np.exp(0 - self.cfg.CLUSTER.STR_GAMMA * campair_temporal_relation)

        self.start_epoch = start_epoch
        for epoch in range(start_epoch, self.cfg.OPTIM.EPOCHS):
            epoch_time = time.time()

            self.generate_pseudo_dataset(epoch)
            self._build_optim(epoch)
            self.init_train()
            self.train(epoch)
            self.eval_save(epoch)

            eta_seconds = (time.time()-epoch_time) * (self.cfg.OPTIM.EPOCHS - (epoch + 1))
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            self.logger.info('eta: {}'.format(eta_str))

    def calculate_smooth_map_K(self, cluster_centers):
        rerank_dist = compute_jaccard_distance(cluster_centers)
        rerank_dist = torch.from_numpy(rerank_dist)
        _, indices = torch.sort(rerank_dist)
        indices_k = indices[:, 0:(self.cfg.MODEL.LOSSES.ID_K + 1)]
        self.nearest_kclusters = torch.zeros_like(rerank_dist)
        for r in range(indices_k.shape[0]):
            for c in range(indices_k.shape[1]):
                self.nearest_kclusters[r][indices_k[r][c]] = 1
        self.nearest_kclusters = self.nearest_kclusters.int().cuda()
    
    def generate_pseudo_dataset(self, epoch):
        self.logger.info('Extract feat and Calculate dist...')
        cluster_sample_multi = self.cfg.DATALOADER.CLUSTER_SAMPLE_MULTI
        num_tracklets = len(self.Target_dataset.train)
        features_map, _, sampled_indexes_map = extract_features(
            self.model_ema, self.Target_cluster_loader, mode="cluster", print_freq=self.cfg.TEST.PRINT_PERIOD,
            sample_multi=cluster_sample_multi)
        cf_multi = [features_map[img_paths] for img_paths, _, _ in sorted(self.Target_dataset.train)]
        cf = torch.stack([
            feat
            for img_paths, _, _ in sorted(self.Target_dataset.train)
            for feat in features_map[img_paths]
        ])
        assert len(cf) == cluster_sample_multi * num_tracklets
        rerank_dist = compute_jaccard_distance(cf)

        if self.cfg.CLUSTER.CCE_OPTION:
            rerank_dist = rerank_dist + self.cross_camera_relation * self.cfg.CLUSTER.CCE_LAMBDA
        
        if self.cfg.CLUSTER.STR_OPTION:
            rerank_dist = rerank_dist * self.spatio_temporal_relation

        if (epoch==0 or epoch == self.start_epoch):
            # DBSCAN cluster
            if self.cfg.CLUSTER.ADAPTIVE:
                tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
                tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
                tri_mat = np.sort(tri_mat,axis=None)
                rho = 1.6e-3
                top_num = np.round(rho*tri_mat.size).astype(int)
                eps = tri_mat[:top_num].mean()
            else:
                eps = self.cfg.CLUSTER.EPS
            self.logger.info('eps for cluster: {:.3f}'.format(eps))
            if self.cfg.CLUSTER.ICDBSCAN_OPTION:
                self.cluster = ICDBSCAN(eps=eps, min_samples=self.cfg.CLUSTER.MIN_SAMPLES, cannotlink_pts=self.icdbscan_cannotlink_pts, relaxed=self.cfg.CLUSTER.ICDBSCAN_RELAXED)
            else:
                self.cluster = DBSCAN(eps=eps, min_samples=self.cfg.CLUSTER.MIN_SAMPLES, metric='precomputed', n_jobs=-1)

        self.logger.info('Clustering and labeling...')
        labels = self.cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        self.num_clusters = num_ids
        self.logger.info('Clustered into {} classes'.format(self.num_clusters))

        assert len(labels) == cluster_sample_multi * num_tracklets
        labels_multi = [
            labels[i * cluster_sample_multi : (i + 1) * cluster_sample_multi]
            for i in range(num_tracklets)
        ]

        ext_size = self.cfg.DATALOADER.SAMPLE_CONSTRAINT_EXT_SIZE
        cons_padding = max(0, ext_size - self.cfg.DATALOADER.CLUSTER_SEQ_LEN) // 2

        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((img_paths, _, cid), label_multi, features) in enumerate(zip(sorted(self.Target_dataset.train), labels_multi, cf_multi)):
            sampled_indexes_multi = sampled_indexes_map[img_paths]
            for feat, sampled_indexes, label in zip(features, sampled_indexes_multi, label_multi):
                if label==-1: continue
                if self.cfg.DATALOADER.SAMPLE_CONSTRAINT:
                    min_idx = max(min(sampled_indexes) - cons_padding, 0)
                    max_idx = min(max(sampled_indexes) + cons_padding, len(img_paths) - 1)
                    new_img_paths = tuple([img_paths[idx] for idx in range(min_idx, max_idx + 1)])
                else:
                    new_img_paths = img_paths
                new_dataset.append((new_img_paths,label,cid))
                cluster_centers[label].append(feat)

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        if self.cfg.MODEL.LOSSES.ID_OPTION == 3:
            self.calculate_smooth_map_K(cluster_centers)
        # if len(self.cfg.GPU_Device) == 1:
        #     self.model.classifier.weight.data[:self.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
        #     self.model_ema.classifier.weight.data[:self.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
        # else:
        self.model.module.classifier.weight.data[:self.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
        self.model_ema.module.classifier.weight.data[:self.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        iters_cluster_multiplier = self.cfg.DATALOADER.ITERS_CLUSTER_MULTIPLIER
        if iters_cluster_multiplier > 0:
            self.num_iters = round(iters_cluster_multiplier * self.num_clusters)
        else:
            self.num_iters = self.cfg.DATALOADER.ITERS

        self.Target_train_loader = build_loader(self.cfg, None, inputset=new_dataset, is_train=True, num_iters=self.num_iters)

        with open(osp.join(self.cfg.OUTPUT_DIR, f"cluster_result_{epoch:03}.pickle"), "wb") as f:
            pickle.dump(new_dataset, f)


    def eval(self):
        return self.evaluator.evaluate(self.Target_test_loader,
                self.Target_dataset.query, self.Target_dataset.gallery)

    def eval_save(self, epoch):
        if (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] > 0 and (epoch+1) not in self.cfg.CHECKPOING.SAVE_STEP:
            return
        elif (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] < 0  and (epoch+1) % -self.cfg.CHECKPOING.SAVE_STEP[0] != 0:
            return

        top1, mAP = self.eval()

        is_top1_best = top1 > self.best_top1
        self.best_top1 = max(top1, self.best_top1)
        is_mAP_best = mAP > self.best_mAP
        self.best_mAP = max(mAP, self.best_mAP)

        save_checkpoint({
            'state_dict': self.model_ema.module.state_dict(),
            'student_dict': self.model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': self.best_top1,
            'best_mAP': self.best_mAP
        }, is_top1_best, is_mAP_best, fpath=osp.join(self.cfg.OUTPUT_DIR, 'checkpoint_new.pth.tar'), remain=self.cfg.CHECKPOING.REMAIN_CLASSIFIER)

        self.logger.info('Finished epoch {:3d}\nTarget mAP: {:5.1%}  best: {:5.1%}{}\nTarget top1: {:5.1%}  best: {:5.1%}{}'.
              format(epoch + 1, mAP, self.best_mAP, ' *' if is_mAP_best else '', top1, self.best_top1, ' *' if is_top1_best else ''))
        return

    def init_train(self):
        if self.cfg.MODEL.LOSSES.ID_OPTION == 1:
            self.criterion_ce = CrossEntropyLoss().cuda()
        elif self.cfg.MODEL.LOSSES.ID_OPTION == 2:
            self.criterion_ce = CrossEntropyLabelSmooth(self.num_clusters, epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        else:
            self.criterion_ce = CrossEntropyLabelSmoothK(self.nearest_kclusters, self.cfg.MODEL.LOSSES.ID_K, epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        
        if self.cfg.MODEL.LOSSES.TRI_OPTION == 1:
            self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        else:
            self.criterion_tri = WeightAdaptiveTripletLoss(self.cfg.MODEL.LOSSES.TRI_MARGIN, self.cfg.MODEL.LOSSES.TRI_LAMBDA).cuda()

        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch):
        self.logger.info('lr: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.Target_train_loader.new_epoch()
        self.optimizer.zero_grad()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        def _forward(inputs, targets):
            f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_clusters]
            f_out_t_ema, p_out_t_ema = self.model_ema(inputs)
            p_out_t_ema = p_out_t_ema[:,:self.num_clusters]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t, p_out_t_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t, f_out_t_ema, targets)

            loss = loss_ce * (1-self.cfg.MUTUAL_TEACH.CE_SOFT_WRIGHT) + \
                    loss_tri * (1-self.cfg.MUTUAL_TEACH.TRI_SOFT_WRIGHT) + \
                    loss_ce_soft * self.cfg.MUTUAL_TEACH.CE_SOFT_WRIGHT + \
                    loss_tri_soft * self.cfg.MUTUAL_TEACH.TRI_SOFT_WRIGHT

            # loss.backward()

            prec = accuracy(p_out_t_ema.data, targets.data)
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions.update(prec)

            return loss

        self.model.train()
        self.model_ema.train()
        for i in range(self.num_iters):
            data_time.update(time.time() - end)

            target_inputs = self.Target_train_loader.next()
            inputs, targets = self._parse_data(target_inputs)

            if len(self.cfg.GPU_Device) == 1:
                n = self.cfg.DATALOADER.TRAIN_BATCH_SIZE // self.cfg.OPTIM.FORWARD_BATCH_SIZE
                loss = torch.tensor(0.0,requires_grad=True).cuda()
                for j in range(n):
                    loss += _forward(inputs[j * self.cfg.OPTIM.FORWARD_BATCH_SIZE : (j+1) * self.cfg.OPTIM.FORWARD_BATCH_SIZE],
                        targets[j * self.cfg.OPTIM.FORWARD_BATCH_SIZE : (j+1) * self.cfg.OPTIM.FORWARD_BATCH_SIZE])

                loss = loss / 4
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_ema_variables(self.model, self.model_ema, self.cfg.MUTUAL_TEACH.ALPHA, epoch*len(self.Target_train_loader) + i)
            else:

                f_out_t, p_out_t = self.model(inputs)
                p_out_t = p_out_t[:,:self.num_clusters]
                f_out_t_ema, p_out_t_ema = self.model_ema(inputs)
                p_out_t_ema = p_out_t_ema[:,:self.num_clusters]

                loss_ce = self.criterion_ce(p_out_t, targets)
                loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)

                loss_ce_soft = self.criterion_ce_soft(p_out_t, p_out_t_ema)
                loss_tri_soft = self.criterion_tri_soft(f_out_t, f_out_t_ema, targets)

                loss = loss_ce * (1-self.cfg.MUTUAL_TEACH.CE_SOFT_WRIGHT) + \
                        loss_tri * (1-self.cfg.MUTUAL_TEACH.TRI_SOFT_WRIGHT) + \
                        loss_ce_soft * self.cfg.MUTUAL_TEACH.CE_SOFT_WRIGHT + \
                        loss_tri_soft * self.cfg.MUTUAL_TEACH.TRI_SOFT_WRIGHT

                prec = accuracy(p_out_t_ema.data, targets.data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_ema_variables(self.model, self.model_ema, self.cfg.MUTUAL_TEACH.ALPHA, epoch*len(self.Target_train_loader) + i)

                losses_ce.update(loss_ce.item())
                losses_tri.update(loss_tri.item())
                losses_ce_soft.update(loss_ce_soft.item())
                losses_tri_soft.update(loss_tri_soft.item())
                precisions.update(prec)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % self.cfg.PRINT_PERIOD == 0) or ((i + 1) % self.num_iters == 0):
                self.logger.info('Epoch: [{}][{}/{}]\t'
                                 'Time {:.3f} ({:.3f})\t'
                                 'Data {:.3f} ({:.3f})\t'
                                 'Loss_ce {:.3f} ({:.3f})\t'
                                 'Loss_tri {:.3f} ({:.3f})\t'
                                 'Loss_ce_soft {:.3f} ({:.3f})\t'
                                 'Loss_tri_soft {:.3f} ({:.3f})\t'
                                 'Prec {:.2%} ({:.2%})\t'
                                 .format(epoch + 1, i + 1, self.num_iters,
                                         batch_time.val, batch_time.avg,
                                         data_time.val, data_time.avg,
                                         losses_ce.val, losses_ce.avg,
                                         losses_tri.val, losses_tri.avg,
                                         losses_ce_soft.val, losses_ce_soft.avg,
                                         losses_tri_soft.val, losses_tri_soft.avg,
                                         precisions.val, precisions.avg))

        # self.lr_scheduler.step()

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)

    def _parse_data(self, inputs):
        imgs, _, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
