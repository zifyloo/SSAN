# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""


from option.options import options, config
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import CRLoss
from torch import optim
import logging
import os
from test_during_train import test
from torch.autograd import Variable


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model') + '/' + str(opt.start_epoch) + '.pth.tar'
    state = torch.load(filename)

    return state


def calculate_similarity(image_embedding, text_embedding):
    image_embedding = image_embedding.view(image_embedding.size(0), -1)
    text_embedding = text_embedding.view(text_embedding.size(0), -1)
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    similarity_match = torch.sum(image_embedding_norm * text_embedding_norm, dim=1)

    return similarity, similarity_match


def calculate_lambda_hr(similarity_matching_hr, similarity_matching):
    lambda_hr = abs(similarity_matching_hr.detach()) / abs(similarity_matching.detach())
    ones = torch.ones_like(lambda_hr)
    data = torch.ge(ones, lambda_hr).float()
    data_2 = torch.ge(lambda_hr, ones).float()
    lambda_hr = data * lambda_hr + data_2

    return lambda_hr


def train(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    config(opt)
    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'

    id_loss_fun_global = Id_Loss(opt, 1, opt.feature_length).to(opt.device)
    id_loss_fun_local = Id_Loss(opt, opt.part, opt.feature_length).to(opt.device)
    id_loss_fun_non_local = Id_Loss(opt, opt.part, 512).to(opt.device)
    cr_loss_fun = CRLoss(opt)
    network = TextImgPersonReidNet(opt).to(opt.device)

    cnn_params = list(map(id, network.ImageExtract.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
    other_params = list(other_params)
    other_params.extend(list(id_loss_fun_global.parameters()))
    other_params.extend(list(id_loss_fun_local.parameters()))
    other_params.extend(list(id_loss_fun_non_local.parameters()))
    param_groups = [{'params': other_params, 'lr': opt.lr},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1}]

    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_history = 0

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    for epoch in range(opt.epoch):

        id_loss_sum = 0
        ranking_loss_sum = 0

        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))

        for times, [image, label, caption_code, caption_length, caption_code_hr, caption_length_hr] in enumerate(
                train_dataloader):

            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))
            caption_code = Variable(caption_code.to(opt.device).long())
            caption_length = caption_length.to(opt.device)
            caption_code_hr = Variable(caption_code_hr.to(opt.device).long())
            caption_length_hr = caption_length_hr.to(opt.device)

            img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local = network(image, caption_code,
                                                                                                 caption_length)

            txt_global_hr, txt_local_hr, txt_non_local_hr = network.txt_embedding(caption_code_hr, caption_length_hr)

            id_loss_global = id_loss_fun_global(img_global, txt_global, label)
            id_loss_local = id_loss_fun_local(img_local, txt_local, label)
            id_loss_non_local = id_loss_fun_non_local(img_non_local, txt_non_local, label)
            id_loss = id_loss_global + (id_loss_local + id_loss_non_local) * 0.5

            cr_loss_global = cr_loss_fun(img_global, txt_global, txt_global_hr, label, epoch >= opt.epoch_begin)
            cr_loss_local = cr_loss_fun(img_local, txt_local, txt_local_hr, label, epoch >= opt.epoch_begin)
            cr_loss_non_local = cr_loss_fun(img_non_local, txt_non_local,
                                            txt_non_local_hr, label, epoch >= opt.epoch_begin)

            ranking_loss = cr_loss_global + (cr_loss_local + cr_loss_non_local) * 0.5

            optimizer.zero_grad()
            loss = (id_loss + ranking_loss)
            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Setp: %d, ranking_loss: %.2f, id_loss: %.2f"
                             % (epoch + 1, opt.epoch, times + 1, ranking_loss, id_loss))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)

        logging.info("Epoch: %d/%d , ranking_loss: %.2f, id_loss: %.2f"
                     % (epoch + 1, opt.epoch, ranking_loss_avg, id_loss_avg))

        print(opt.model_name)
        network.eval()
        test_best = test(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best)
        network.train()

        if test_best > test_history:
            test_history = test_best
            state = {
                'network': network.cpu().state_dict(),
                'test_best': test_best,
                'epoch': epoch,
                'WN': id_loss_fun_non_local.cpu().state_dict(),
                'WL': id_loss_fun_local.cpu().state_dict(),
            }
            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun_non_local.to(opt.device)
            id_loss_fun_local.to(opt.device)

        scheduler.step()

    logging.info('Training Done')


if __name__ == '__main__':
    opt = options().opt
    train(opt)





