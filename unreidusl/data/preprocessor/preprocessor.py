from __future__ import absolute_import
import os
import os.path as osp
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
            return img
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

class Preprocessor(Dataset):
    """Video Person ReID Dataset.
    Note __getitem__ return data has shape (seq_len, channel, height, width).
    """
    sample_methods = ['dense', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='all', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        img_paths_joined = "$".join(img_paths)
        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            imgs, sampled_indexes = self.random_sample(img_paths)
            return imgs, "#".join([str(x) for x in sampled_indexes]), img_paths_joined, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            return self.dense_sample(img_paths), "$".join(img_paths), pid, camid
        
        elif self.sample == 'all':
            """
            Sample all frames in a video, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            return self.all_sample(img_paths), "#".join([str(x) for x in range(num)]), "$".join(img_paths), pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

    def random_sample(self, img_paths):
        frame_indices = list(range(len(img_paths)))
        rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.seq_len, len(frame_indices))

        indices = frame_indices[begin_index:end_index]
        sampled_indices = [*indices]

        for index in indices:
            if len(indices) >= self.seq_len:
                break
            indices.append(index)
        indices = np.array(indices)
        imgs = []
        for index in indices:
            index = int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs, sampled_indices

    def all_sample(self, img_paths):
        imgs = []
        for img_path in img_paths:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def dense_sample(self, img_paths):
        cur_index=0
        frame_indices = list(range(len(img_paths)))
        indices_list=[]
        while len(img_paths)-cur_index > self.seq_len:
            indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
            cur_index+=self.seq_len
        last_seq=frame_indices[cur_index:]
        for index in last_seq:
            if len(last_seq) >= self.seq_len:
                break
            last_seq.append(index)
        indices_list.append(last_seq)
        imgs_list=[]
        for indices in indices_list:
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            imgs_list.append(imgs)
        imgs_array = torch.stack(imgs_list)
        return imgs_array

