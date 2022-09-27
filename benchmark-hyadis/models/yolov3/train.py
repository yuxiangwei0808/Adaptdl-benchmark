import time
import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import os
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler

from torch.optim.lr_scheduler import CosineAnnealingLR
torch.autograd.set_detect_anomaly(True)

class Trainer(object):
    def __init__(self, weight_path):
        init_seeds(0)
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                 batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                 num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                                                 drop_last=True,
                                                                 shuffle=True)
        self.valid_dataset = data.VocDataset(anno_file_type="test")
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                                 batch_size=8,
                                                                 num_workers=8,
                                                                 shuffle=False)
        self.yolov3 = Yolov3().cuda()

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.yolov3.load_darknet_weights(weight_path)

        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=(self.epochs - cfg.TRAIN["WARMUP_EPOCHS"]),
                                           eta_min=cfg.TRAIN["LR_END"])

    def valid(self, epoch, writer):
        self.yolov3.train()

        with torch.no_grad():
            for imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes in self.valid_dataloader:
                imgs = imgs.cuda()
                label_sbbox = label_sbbox.cuda()
                label_mbbox = label_mbbox.cuda()
                label_lbbox = label_lbbox.cuda()
                sbboxes = sbboxes.cuda()
                mbboxes = mbboxes.cuda()
                lbboxes = lbboxes.cuda()

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

                # Print batch results
                print("Epoch {} valid [{}/{}]:  loss_giou: {:.4f}  loss_conf: {:.4f}  loss_cls: {:.4f}  loss: {:.4f}"
                      .format(epoch, self.valid_dataloader._elastic.current_index, len(self.valid_dataset),
                              loss_giou.item(), loss_conf.item(), loss_cls.item(), loss.item()))

                del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                del p, p_d, loss, loss_giou, loss_conf, loss_cls



    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in range(self.epochs):
            self.yolov3.train()
            for idx, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):
                imgs = imgs.cuda()
                label_sbbox = label_sbbox.cuda()
                label_mbbox = label_mbbox.cuda()
                label_lbbox = label_lbbox.cuda()
                sbboxes = sbboxes.cuda()
                mbboxes = mbboxes.cuda()
                lbboxes = lbboxes.cuda()

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

                loss.backward()
                self.optimizer.step()

                # Multi-scale training (320-608 pixels).
                if self.multi_scale_train:
                    self.train_dataset.img_size = random.choice(range(10, 20)) * 32

                del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                del p, p_d, loss, loss_giou, loss_conf, loss_cls

                if epoch < cfg.TRAIN["WARMUP_EPOCHS"]:
                    for group in self.optimizer.param_groups:
                        group["lr"] = (idx / len(self.train_dataset) *
                                    self.train_dataloader.batch_size /
                                    cfg.TRAIN["WARMUP_EPOCHS"]) * cfg.TRAIN["LR_INIT"]
                print("lr =", self.optimizer.param_groups[0]["lr"])

            if epoch >= cfg.TRAIN["WARMUP_EPOCHS"]:
                self.scheduler.step()


if __name__ == "__main__":
    s = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path).train()