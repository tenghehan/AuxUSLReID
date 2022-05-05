
import json
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

from .base_dataset import BaseVideoDataset
from ...utils.defaults import read_json, write_json
from ...config import get_cfg

class DukeMTMCVidReID(BaseVideoDataset):
    """
    DukeMTMCVidReID
    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.
    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)

    Dataset statistics:
    ------------------------------
    subset         | # ids | # tracklets | # images
    ---------------------------------------
    train          |   702 |     2196 |   369656
    train_dense    |   702 |    10517 |   369656
    query          |   702 |      702 |   111848
    gallery        |  1110 |     2636 |   445764
    ---------------------------------------
    total          |  1404 |     5534 |   927268
    number of images per tracklet: 1 ~ 9324, average 167.6
    number of images per tracklet (train): 5 ~ 7507, average 168.3
    ---------------------------------------
    """

    def __init__(self, root, cfg, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, cfg.DATASETS.NAME)
        # self.dataset_dir = osp.join(root, "DukeMTMC-VideoReID")
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self.split_train_json_path = osp.join(self.dataset_dir, "split_train.json")
        self.split_train_dense_json_path = osp.join(self.dataset_dir, "split_train_dense_{}.json".format(cfg.DATASETS.DENSE_STEP))
        self.split_query_json_path = osp.join(self.dataset_dir, "split_query.json")
        self.split_gallery_json_path = osp.join(self.dataset_dir, "split_gallery.json")

        self.min_seq_len = min_seq_len

        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        train_dense, num_train_tracklets_dense, num_train_pids_dense, num_imgs_train_dense = \
          self._process_dir_dense(self.train_dir, self.split_train_dense_json_path, relabel=True, sampling_step=cfg.DATASETS.DENSE_STEP)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        print("the number of tracklets under dense sampling for train set: {}".format(num_train_tracklets_dense))

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset         | # ids | # tracklets | # images")
            print("  ---------------------------------------")
            print("  train          | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets, np.sum(num_imgs_train)))
            print("  train_dense    | {:5d} | {:8d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense, np.sum(num_imgs_train_dense)))
            print("  query          | {:5d} | {:8d} | {:8d}".format(num_query_pids, num_query_tracklets, np.sum(num_imgs_query)))
            print("  gallery        | {:5d} | {:8d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, np.sum(num_imgs_gallery)))
            print("  ---------------------------------------")
            print("  total          | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets, np.sum(num_imgs_per_tracklet)))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ---------------------------------------")

        if cfg.DATASETS.MODE == "dense":
            self.train = train_dense
        else:
            self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _list_to_tuple(self, tracklets):
        new_tracklets = []
        for img_paths, pid, camid in tracklets:
            new_tracklets.append((tuple(img_paths), pid, camid))
        return new_tracklets

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return self._list_to_tuple(split['tracklets']), split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, "*"))
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in tqdm(pdirs):
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, "*"))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, "*.jpg"))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []

                for img_idx in range(num_imgs):
                    img_idx_name = "F" + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, "*" + img_idx_name + "*.jpg"))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])

                img_name = osp.basename(img_paths[0])
                camid = int(img_name[6])

                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": num_pids,
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
        

    def _process_dir_dense(self, dir_path, json_path, relabel, sampling_step=32):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return self._list_to_tuple(split['tracklets']), split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, "*"))
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in tqdm(pdirs):
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, "*"))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, "*.jpg"))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []

                for img_idx in range(num_imgs):
                    img_idx_name = "F" + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, "*" + img_idx_name + "*.jpg"))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])

                img_name = osp.basename(img_paths[0])
                camid = int(img_name[6])

                img_paths = tuple(img_paths)

                # dense sampling
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))
                            
        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": num_pids,
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


class DukeMTMCTrack(BaseVideoDataset):
    def __init__(self, root, cfg, min_seq_len=0, verbose=True, **kwargs):
        """
        Dataset statistics:
        ------------------------------
        subset         | # ids | # tracklets | # images
        ---------------------------------------
        train          |  4002 |     4002 |   344544
        query          |   702 |      702 |   111848
        gallery        |  1110 |     2636 |   445764
        ---------------------------------------
        total          |  4704 |     7340 |   902156
        number of images per tracklet: 1 ~ 9324, average 122.9
        ---------------------------------------
        """
        self.dataset_dir = osp.join(root, cfg.DATASETS.NAME)
        self.info_dir = cfg.DATASETS.DUKE_INFO
        self.train_name_path = osp.join(self.dataset_dir, self.info_dir, "train_name.txt")
        self.track_train_info_path = osp.join(self.dataset_dir, self.info_dir, "tracks_train_info.mat")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self.split_train_json_path = osp.join(self.dataset_dir, "split_train.json")
        self.split_train_dense_json_path = osp.join(self.dataset_dir, "split_train_duketrack_dense_{}.json".format(cfg.DATASETS.DENSE_STEP))
        self.split_query_json_path = osp.join(self.dataset_dir, "split_query.json")
        self.split_gallery_json_path = osp.join(self.dataset_dir, "split_gallery.json")

        self.min_seq_len = min_seq_len

        self._check_before_run()

        train_names = self._get_names(self.train_name_path)
        track_train = loadmat(self.track_train_info_path)["track_train_info"]

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_dir_train(self.split_train_json_path, train_names, track_train, home_dir="bbox_train", relabel=True, min_seq_len=self.min_seq_len)
        train_dense, num_train_dense_tracklets, num_train_dense_pids, num_imgs_train_dense = \
            self._process_dir_train_dense(self.split_train_dense_json_path, train_names, track_train, home_dir="bbox_train", relabel=True, min_seq_len=self.min_seq_len, sampling_step=cfg.DATASETS.DENSE_STEP)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_dir_test(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_dir_test(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset         | # ids | # tracklets | # images")
            print("  ---------------------------------------")
            print("  train          | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets, np.sum(num_imgs_train)))
            print("  train_dense    | {:5d} | {:8d} | {:8d}".format(num_train_dense_pids, num_train_dense_tracklets, np.sum(num_imgs_train_dense)))
            print("  query          | {:5d} | {:8d} | {:8d}".format(num_query_pids, num_query_tracklets, np.sum(num_imgs_query)))
            print("  gallery        | {:5d} | {:8d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, np.sum(num_imgs_gallery)))
            print("  ---------------------------------------")
            print("  total          | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets, np.sum(num_imgs_per_tracklet)))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ---------------------------------------")

        if cfg.DATASETS.MODE == "dense":
            self.train = train_dense
        else:
            self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _list_to_tuple(self, tracklets):
        new_tracklets = []
        for img_paths, pid, camid in tracklets:
            new_tracklets.append((tuple(img_paths), pid, camid))
        return new_tracklets

    def _process_dir_train(self, json_path, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return self._list_to_tuple(split['tracklets']), split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        assert home_dir == 'bbox_train'
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 8
            if relabel: pid = pid2label[pid]
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            # pnames = [img_name[:4] for img_name in img_names]
            # if self.info_dir == "info" or self.info_dir == "info_x2":
            #     assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, int(pid), int(camid)))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)



        print("Saving split to {}".format(json_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": len(pid_list),
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, len(pid_list), num_imgs_per_tracklet

    def _process_dir_train_dense(self, json_path, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, sampling_step=32):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return self._list_to_tuple(split['tracklets']), split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        assert home_dir == 'bbox_train'
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 8
            if relabel: pid = pid2label[pid]
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            # pnames = [img_name[:4] for img_name in img_names]
            # if self.info_dir == "info" or self.info_dir == "info_x2":
            #     assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)

                # dense sampling
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, int(pid), int(camid)))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], int(pid), int(camid)))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], int(pid), int(camid)))
          
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)



        print("Saving split to {}".format(json_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": len(pid_list),
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, len(pid_list), num_imgs_per_tracklet

    def _process_dir_test(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return self._list_to_tuple(split['tracklets']), split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, "*"))
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in tqdm(pdirs):
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, "*"))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, "*.jpg"))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []

                for img_idx in range(num_imgs):
                    img_idx_name = "F" + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, "*" + img_idx_name + "*.jpg"))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])

                img_name = osp.basename(img_paths[0])
                camid = int(img_name[6])

                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": num_pids,
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


if __name__ == "__main__":
    cfg = get_cfg()
    # cfg.DATASETS.NAME = "DukeMTMC-VideoReID"
    # cfg.DATASETS.DENSE_STEP = 56
    # dataset = DukeMTMCVidReID(root="datasets", cfg=cfg)

    cfg.DATASETS.NAME = "DukeMTMC-Track"
    cfg.DATASETS.DUKE_INFO = "info"
    cfg.DATASETS.MODE = "dense"
    cfg.DATASETS.DENSE_STEP = 48
    dataset = DukeMTMCTrack(root="datasets", cfg=cfg)