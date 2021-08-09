# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""


from option.options import options
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
import os
from test_during_train import test


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    state = torch.load(filename)
    print('Load the {} epoch parameter successfully'.format(state['epoch']))

    return state


def main(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)

    network = TextImgPersonReidNet(opt).to(opt.device)

    test_best = 0
    state = load_checkpoint(opt)
    network.load_state_dict(state['network'])
    epoch = state['epoch']

    print(opt.model_name)
    network.eval()
    test(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best, False)
    network.train()


if __name__ == '__main__':
    opt = options().opt
    main(opt)





