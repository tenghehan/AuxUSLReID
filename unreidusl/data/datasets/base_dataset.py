# encoding: utf-8
import numpy as np
import logging

class BaseDataset(object):
    """
    Base class of video reid dataset
    """

    def get_videodata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseVideoDataset(BaseDataset):
    """
    Base class of video reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams = self.get_videodata_info(train)
        num_query_pids, num_query_tracklets, num_query_cams = self.get_videodata_info(query)
        num_gallery_pids, num_gallery_tracklets, num_gallery_cams = self.get_videodata_info(gallery)

        logger = logging.getLogger('UnReID')
        logger.info("Dataset statistics:" +
                    "\n  ----------------------------------------" +
                    "\n  subset   | # ids | # tracklets | # cameras" +
                    "\n  ----------------------------------------" +
                    "\n  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams) +
                    "\n  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams) +
                    "\n  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams) +
                    "\n  ----------------------------------------")
