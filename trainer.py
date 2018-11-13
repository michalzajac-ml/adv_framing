import os
import random

import torch.nn as nn
import torch.optim
import torch.utils.data

from imagenet import ImageNet
from logger import Logger
from ucf101 import UCF101
from utils import accuracy, get_current_datetime, AverageMeter


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not args.run_id:
            args.run_id = '{}-{}'.format(args.dataset, get_current_datetime())
        self.checkpoint_path = os.path.join(args.checkpoint_dir, args.run_id) + '.chk'
        self.logger = Logger(tb_dir=os.path.join(args.tb_dir, args.run_id),
                             log_path=os.path.join(args.log_dir, args.run_id) + '.log')
        self.logger.log_str(str(args))
        self.args = args

        self.target = args.target

        dataset = {'imagenet': ImageNet, 'ucf101': UCF101}[args.dataset]

        self.classifier = dataset.get_classifier().to(self.device)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.framing = dataset.get_framing(args.width, keep_size=args.keep_size).to(self.device)

        self.train_loader, self.val_loader = dataset.get_data_loaders(args.batch_size, args.workers)
        if self.target == -1:
            self.target = random.randint(0, dataset.NUM_CLASSES - 1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.Adam(self.framing.parameters(), args.lr)
        self.step = 0

    def adjust_learning_rate(self, epoch):
        if epoch % self.args.lr_decay_wait == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.args.lr_decay_coefficient

    def process_epoch(self, train):
        if train:
            data_loader = self.train_loader
        else:
            data_loader = self.val_loader

        loss_unatt_agg = AverageMeter()
        acc_unatt_agg = AverageMeter()
        loss_att_agg = AverageMeter()
        acc_att_agg = AverageMeter()

        for i, (input, target) in enumerate(data_loader):
            if self.target:
                target = self.target * target.new_ones(target.size())
            input, target = input.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.classifier(input)
                los_unatt = self.criterion(output, target)
            loss_unatt_agg.update(los_unatt.item(), input.size(0))
            acc_unatt_agg.update(accuracy(output, target).item(), input.size(0))

            with torch.set_grad_enabled(train):
                input_att, _ = self.framing(input=input)
                output_att = self.classifier(input_att)
                loss_att = self.criterion(output_att, target)
            loss_att_agg.update(loss_att.item(), input.size(0))
            acc_att_agg.update(accuracy(output_att, target).item(), input.size(0))

            if train:
                self.optimizer.zero_grad()
                framing_loss = loss_att if self.target is not None else -loss_att
                framing_loss.backward()
                self.optimizer.step()
                self.step += input.size(0)

            if train:
                if (i + 1) % self.args.print_freq == 0:
                    self.logger.log_kv([
                        ('unatt_loss', loss_unatt_agg.avg),
                        ('att_loss', loss_att_agg.avg),
                        ('unatt_acc', acc_unatt_agg.avg),
                        ('att_acc', acc_att_agg.avg),
                    ], prefix='train', step=self.step, write_to_tb=True)
                    loss_unatt_agg.reset()
                    loss_att_agg.reset()
                    acc_unatt_agg.reset()
                    acc_att_agg.reset()
            else:
                if i + 1 == len(data_loader):
                    self.logger.log_kv([
                        ('unatt_loss', loss_unatt_agg.avg),
                        ('att_loss', loss_att_agg.avg),
                        ('unatt_acc', acc_unatt_agg.avg),
                        ('att_acc', acc_att_agg.avg),
                    ], prefix='eval', step=self.step, write_to_tb=True, write_to_file=True)

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.process_epoch(train=True)
            torch.save({
                'epoch': epoch,
                'framing': self.framing.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'args': self.args,
            }, self.checkpoint_path)
            self.process_epoch(train=False)
            self.adjust_learning_rate(epoch)
