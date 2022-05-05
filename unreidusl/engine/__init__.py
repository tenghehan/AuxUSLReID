from __future__ import absolute_import

from .usl_MTCluster_base import usl_MTCluster_base

__factory = {
    'usl_MTCluster_base': usl_MTCluster_base
}

def names():
    return sorted(__factory.keys())

def create_engine(cfg):
    if cfg.MODE not in __factory:
        raise KeyError("Unknown Engine:", cfg.MODE)
    engine = __factory[cfg.MODE](cfg)
    return engine
