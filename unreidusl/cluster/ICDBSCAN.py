import numpy as np
import copy
from collections import deque

UNCLASSIFIED = -2
NOISE = -1

class ICDBSCAN:
    def __init__(self, eps=3, min_samples=2, cannotlink_pts=[], relaxed=1.0):
        self.eps = eps
        self.min_samples = min_samples
        self.cannotlink_pts = cannotlink_pts
        self.relaxed = relaxed

    def initBeforePredict(self):
        self.clusterId = 0
        self.neighbors = {}
        self.relaxedNeighbors = {}
        self.disMat = None
        self.num = 0
        self.clusterCount = 0
        self.assignCluster = None

    def computeNeighbor(self):
        for idx in range(self.num):
            neighs = set(deque([i for i, x in enumerate(self.disMat[idx] <= self.eps) if x and i != idx]))
            neighs.add(idx)
            self.neighbors[idx] = neighs

    def relaxNeighbor(self):
        if self.relaxed >= 0.0:
            for idx in range(self.num):
                neighs = set(deque([i for i, x in enumerate(self.disMat[idx] <= (self.eps * self.relaxed)) if x and i != idx]))
                neighs.add(idx)
                self.relaxedNeighbors[idx] = neighs
        else:
            for idx in range(self.num):
                neighs = set(deque([i for i, x in enumerate(self.disMat[idx] > (self.eps * (0 - self.relaxed))) if x and i != idx]))
                neighs.add(idx)
                self.relaxedNeighbors[idx] = neighs

        for idx, neighs in self.relaxedNeighbors.items():
            self.cannotlink_pts[idx] = self.cannotlink_pts[idx] - neighs

    def fit_predict(self, disMat):
        self.initBeforePredict()
        self.disMat = disMat
        self.num = disMat.shape[0]
        self.computeNeighbor()
        self.relaxNeighbor()
        return self.run()

    def run(self):
        self.assignCluster = UNCLASSIFIED * np.ones(self.num, dtype=np.int64)
        curClusterId = self.clusterId

        D = set([i for i in range(self.num)])
        for pt in D:
            if self.assignCluster[pt] == UNCLASSIFIED:
                epr = self.ExpandCluster(pt, curClusterId)
                if epr:
                    curClusterId += 1
                    self.clusterCount += 1

        for i in range(self.num):
            if self.assignCluster[i] == UNCLASSIFIED:
                self.assignCluster[i] = NOISE

        return self.assignCluster

    def ExpandCluster(self, pid, curClusterId):
        temp = set()
        seeds = set()

        seeds.add(pid)
        temp.add(pid)

        self.assignCluster[pid] = curClusterId
        cannotlinks = self.cannotlink_pts[pid]

        while(len(seeds) > 0):
            curPt = list(seeds)[0]
            curSeeds = self.neighbors[curPt]
            curSeeds = curSeeds - self.cannotlink_pts[curPt] - cannotlinks

            if len(curSeeds) >= self.min_samples:
                for qid in curSeeds:
                    if self.assignCluster[qid] == UNCLASSIFIED and qid not in cannotlinks:
                        self.assignCluster[qid] = curClusterId
                        temp.add(qid)
                        cannotlinks.update(self.cannotlink_pts[qid])
                        seeds.add(qid)

            seeds.remove(curPt)

        if len(temp) < self.min_samples:
            for qid in temp:
                self.assignCluster[qid] = UNCLASSIFIED
            return False

        return True
        