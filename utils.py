from time import gmtime, strftime

import torch


# class copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        return 100. * (predicted == target).sum().float() / output.shape[0]


def get_current_datetime():
    return strftime("%y-%m-%d-%H-%M-%S", gmtime())
