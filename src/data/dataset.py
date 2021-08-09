# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
from utils.read_write_data import read_dict
import cv2
import torchvision.transforms.functional as F
import random


def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip


class CUHKPEDEDataset(data.Dataset):
    def __init__(self, opt, tran):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.caption_code = data_save['lstm_caption_id']

        self.same_id_index = data_save['same_id_index']

        self.transform = tran

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)
        label = torch.from_numpy(np.array([self.label[index]])).long()
        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        same_id_index = np.random.randint(len(self.same_id_index[index]))
        same_id_index = self.same_id_index[index][same_id_index]
        same_id_caption_code, same_id_caption_length = self.caption_mask(self.caption_code[same_id_index])

        return image, label, caption_code, caption_length, same_id_caption_code, same_id_caption_length

    def get_data(self, index, img=True):
        if img:
            image = Image.open(self.img_path[index])
            image = self.transform(image)
        else:
            image = 0

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        return image, label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).long()

        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length).long()
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt, tran):

        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.transform = tran

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class CUHKPEDE_txt_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.label = data_save['caption_label']
        self.caption_code = data_save['lstm_caption_id']

        self.num_data = len(self.caption_code)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])
        return label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data





