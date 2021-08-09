# -*- coding: utf-8 -*-
"""

@author: zifyloo
"""

from torch import nn
from model.text_feature_extract import TextExtract
from torchvision import models
import torch
from torch.nn import init
from torch.nn import functional as F


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x


class NonLocalNet(nn.Module):
    def __init__(self, opt, dim_cut=8):
        super(NonLocalNet, self).__init__()
        self.opt = opt

        up_dim_conv = []
        part_sim_conv = []
        cur_sim_conv = []
        conv_local_att = []
        for i in range(opt.part):
            up_dim_conv.append(conv(opt.feature_length//dim_cut, 1024, relu=True, BN=True))
            part_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            cur_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            conv_local_att.append(conv(opt.feature_length, 512))

        self.up_dim_conv = nn.Sequential(*up_dim_conv)
        self.part_sim_conv = nn.Sequential(*part_sim_conv)
        self.cur_sim_conv = nn.Sequential(*cur_sim_conv)
        self.conv_local_att = nn.Sequential(*conv_local_att)

        self.zero_eye = (torch.eye(opt.part, opt.part) * -1e6).unsqueeze(0).to(opt.device)

        self.lambda_softmax = 1

    def forward(self, embedding):
        embedding = embedding.unsqueeze(3)
        embedding_part_sim = []
        embedding_cur_sim = []

        for i in range(self.opt.part):
            embedding_i = embedding[:, :, i, :].unsqueeze(2)

            embedding_part_sim_i = self.part_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_part_sim.append(embedding_part_sim_i)

            embedding_cur_sim_i = self.cur_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_cur_sim.append(embedding_cur_sim_i)

        embedding_part_sim = torch.cat(embedding_part_sim, dim=2)
        embedding_cur_sim = torch.cat(embedding_cur_sim, dim=2)

        embedding_part_sim_norm = l2norm(embedding_part_sim, dim=1)  # N*D*n
        embedding_cur_sim_norm = l2norm(embedding_cur_sim, dim=1)  # N*D*n
        self_att = torch.bmm(embedding_part_sim_norm.transpose(1, 2), embedding_cur_sim_norm)  # N*n*n
        self_att = self_att + self.zero_eye.repeat(self_att.size(0), 1, 1)
        self_att = F.softmax(self_att * self.lambda_softmax, dim=1)  # .transpose(1, 2).contiguous()
        embedding_att = torch.bmm(embedding_part_sim_norm, self_att).unsqueeze(3)

        embedding_att_up_dim = []
        for i in range(self.opt.part):
            embedding_att_up_dim_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_up_dim_i = self.up_dim_conv[i](embedding_att_up_dim_i).unsqueeze(2)
            embedding_att_up_dim.append(embedding_att_up_dim_i)
        embedding_att_up_dim = torch.cat(embedding_att_up_dim, dim=2).unsqueeze(3)

        embedding_att = embedding + embedding_att_up_dim

        embedding_local_att = []
        for i in range(self.opt.part):
            embedding_att_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_i = self.conv_local_att[i](embedding_att_i).unsqueeze(2)
            embedding_local_att.append(embedding_att_i)

        embedding_local_att = torch.cat(embedding_local_att, 2)

        return embedding_local_att.squeeze()


class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        resnet50 = models.resnet50(pretrained=True)
        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.TextExtract = TextExtract(opt)

        self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.local_avgpool = nn.AdaptiveMaxPool2d((opt.part, 1))

        conv_local = []
        for i in range(opt.part):
            conv_local.append(conv(2048, opt.feature_length))
        self.conv_local = nn.Sequential(*conv_local)

        self.conv_global = conv(2048, opt.feature_length)

        self.non_local_net = NonLocalNet(opt, dim_cut=2)
        self.leaky_relu = nn.LeakyReLU(0.25, inplace=True)

        self.conv_word_classifier = nn.Sequential(
            nn.Conv2d(2048, 6, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, caption_id, text_length):

        img_global, img_local, img_non_local = self.img_embedding(image)
        txt_global, txt_local, txt_non_local = self.txt_embedding(caption_id, text_length)

        return img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local

    def img_embedding(self, image):

        image_feature = self.ImageExtract(image)

        image_feature_global = self.global_avgpool(image_feature)
        image_global = self.conv_global(image_feature_global).unsqueeze(2)

        image_feature_local = self.local_avgpool(image_feature)
        image_local = []
        for i in range(self.opt.part):
            image_feature_local_i = image_feature_local[:, :, i, :]
            image_feature_local_i = image_feature_local_i.unsqueeze(2)
            image_embedding_local_i = self.conv_local[i](image_feature_local_i).unsqueeze(2)
            image_local.append(image_embedding_local_i)

        image_local = torch.cat(image_local, 2)

        image_non_local = self.leaky_relu(image_local)
        image_non_local = self.non_local_net(image_non_local)

        return image_global, image_local, image_non_local

    def txt_embedding(self, caption_id, text_length):

        text_feature_g, text_feature_l = self.TextExtract(caption_id, text_length)

        text_global, _ = torch.max(text_feature_g, dim=2, keepdim=True)
        text_global = self.conv_global(text_global).unsqueeze(2)

        text_feature_local = []
        for text_i in range(text_feature_l.size(0)):
            text_feature_local_i = text_feature_l[text_i, :, :text_length[text_i]].unsqueeze(0)

            word_classifier_score_i = self.conv_word_classifier(text_feature_local_i)

            word_classifier_score_i = word_classifier_score_i.permute(0, 3, 2, 1).contiguous()
            text_feature_local_i = text_feature_local_i.repeat(1, 1, 1, 6).contiguous()

            text_feature_local_i = text_feature_local_i * word_classifier_score_i

            text_feature_local_i, _ = torch.max(text_feature_local_i, dim=2)

            text_feature_local.append(text_feature_local_i)

        text_feature_local = torch.cat(text_feature_local, dim=0)

        text_local = []
        for p in range(self.opt.part):
            text_feature_local_conv_p = text_feature_local[:, :, p].unsqueeze(2).unsqueeze(2)
            text_feature_local_conv_p = self.conv_local[p](text_feature_local_conv_p).unsqueeze(2)
            text_local.append(text_feature_local_conv_p)
        text_local = torch.cat(text_local, dim=2)

        text_non_local = self.leaky_relu(text_local)
        text_non_local = self.non_local_net(text_non_local)

        return text_global, text_local, text_non_local

