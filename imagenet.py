import json
import os

import torch
import torchvision
import torchvision.transforms as transforms

from framing import Framing
from resnet import resnet50

with open(os.path.join(os.path.dirname(__file__), 'assets/imagenet_class_index.json')) as f:
    IMAGE_NET_CLASSES_JSON = json.load(f)


class ImageNet:
    MEAN = (0.485, 0.456, 0.406)
    STD_DEV = (0.229, 0.224, 0.225)
    NUM_CLASSES = 1000
    SIDE_LENGTH = 224
    ID_TO_CLASS = [IMAGE_NET_CLASSES_JSON[str(k)][1] for k in range(len(IMAGE_NET_CLASSES_JSON))]
    normalize = transforms.Normalize(MEAN, STD_DEV)

    @staticmethod
    def get_classifier():
        return resnet50(pretrained=True)

    @staticmethod
    def get_framing(width, keep_size=False):
        return Framing(width=width, image_side_length=ImageNet.SIDE_LENGTH, normalize=ImageNet.normalize_not_in_place,
                       scale=1., keep_size=keep_size)

    @staticmethod
    def get_data_loaders(batch_size, num_workers, normalize=True, shuffle_val=False):
        train_dir = os.path.join(os.environ['IMAGENET_DATA_DIR'], 'train')
        val_dir = os.path.join(os.environ['IMAGENET_DATA_DIR'], 'val')

        train_transforms = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        if normalize:
            train_transforms.append(ImageNet.normalize)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(train_dir, transforms.Compose(train_transforms)),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, sampler=None)

        val_transforms = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        if normalize:
            val_transforms.append(ImageNet.normalize)
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(val_dir, transforms.Compose(val_transforms)),
            batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

    @staticmethod
    def normalize_not_in_place(input):
        params_shape = list(input.size())
        for i in range(1, len(params_shape)):
            params_shape[i] = 1
        mean = input.new_tensor(ImageNet.MEAN).view(params_shape)
        std = input.new_tensor(ImageNet.STD_DEV).view(params_shape)
        return (input - mean) / std


def load_pretrained_imagenet_framing(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args = checkpoint['args']
    framing = ImageNet.get_framing(args.width, keep_size=getattr(args, 'keep_size', False))
    framing.load_state_dict(checkpoint['framing'])
    framing.eval()
    return framing
