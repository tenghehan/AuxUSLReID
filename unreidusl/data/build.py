from __future__ import absolute_import
import os.path as osp
from typing import Optional
from .datasets import *
from .samplers import *
from .preprocessor import *
from .transforms import build_transforms
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from unittest import mock
from torch._six import container_abcs

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

def build_data(name, data_dir, cfg):
    # root = osp.join(data_dir, name)
    root = data_dir
    dataset = create(name, root, cfg)
    return dataset

def build_loader(cfg, dataset, inputset=None, is_train=True, *, num_iters: int = 0):

    transformer = build_transforms(cfg, is_train)

    if is_train:
        dataset = sorted(dataset.train) if inputset is None else sorted(inputset)
        sampler = RandomMultipleGallerySampler(dataset, cfg.DATALOADER.NUM_INSTANCE)
        loader = DataLoader(
            Preprocessor(dataset, seq_len=cfg.DATALOADER.TRAIN_SEQ_LEN, sample='random', transform=transformer),
                batch_size=cfg.DATALOADER.TRAIN_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, sampler=sampler,
                shuffle=False, pin_memory=True, drop_last=True)
        if cfg.DATALOADER.ITER_MODE:
            loader = IterLoader(loader, length = num_iters)
    elif inputset is None:
        dataset = list(set(dataset.query) | set(dataset.gallery))
        loader = DataLoader(
            Preprocessor(dataset, sample='all', transform=transformer),
            batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=False, pin_memory=True)
    else:
        dataset = sorted(inputset)
        loader = DataLoader(
            Preprocessor(dataset, seq_len=cfg.DATALOADER.CLUSTER_SEQ_LEN, sample='random', transform=transformer),
            batch_size=cfg.DATALOADER.CLUSTER_BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=False, pin_memory=True)

    return loader

if __name__ == '__main__':
    import logging

    from unreidusl.config.config import get_cfg
    from unreidusl.utils.defaults import default_setup

    logging.basicConfig(level=logging.INFO)
    Target_dataset = build_data('mars', 'datasets')

    cfg = get_cfg()
    cfg.merge_from_file('configs/USL_MTCluster.yml')
    cfg.freeze()

    Target_cluster_loader = build_loader(cfg, None, inputset=sorted(Target_dataset.train), is_train=False)
    Target_test_loader = build_loader(cfg, Target_dataset, is_train=False)
    Target_train_loader = build_loader(cfg, None, inputset=sorted(Target_dataset.train), is_train=True)

    print(len(Target_cluster_loader))
    print(len(Target_test_loader))
    print(len(Target_train_loader))

    Target_train_loader.new_epoch()
    x = Target_train_loader.next()
    print(x)
