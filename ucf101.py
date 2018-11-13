import os

import torch
from bunch import Bunch

import resnets_3d.dataset
from framing import Framing
from resnets_3d import spatial_transforms, temporal_transforms, target_transforms
from resnets_3d.mean import get_mean
from resnets_3d.model import generate_model


def get_ucf101_train_data_loader(batch_size, num_workers, data_dir, annotation_path):
    norm_value = 1
    mean = get_mean(norm_value, dataset='kinetics')
    normalize = spatial_transforms.Normalize(mean, [1, 1, 1])
    scales = [1.0]
    for i in range(1, 5):
        scales.append(scales[-1] * 0.84089641525)
    spatial_transform = spatial_transforms.Compose([
        spatial_transforms.MultiScaleCornerCrop(scales, UCF101.SIDE_LENGTH),
        spatial_transforms.RandomHorizontalFlip(),
        spatial_transforms.ToTensor(norm_value), normalize
    ])
    temporal_transform = temporal_transforms.TemporalRandomCrop(16)
    target_transform = target_transforms.ClassLabel()
    opt = Bunch(
        dataset='ucf101',
        video_path=data_dir,
        annotation_path=annotation_path,
    )
    training_data = resnets_3d.dataset.get_training_set(opt, spatial_transform,
                                                        temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    return train_loader, training_data.class_names


def get_ucf101_val_data_loader(batch_size, num_workers, data_dir, annotation_path, shuffle=False, normalize=True,
                               whole_videos=False):
    norm_value = 1
    mean = get_mean(norm_value, dataset='kinetics')
    normalization_transform = spatial_transforms.Normalize(mean, [1, 1, 1])
    transforms = [
        spatial_transforms.Scale(UCF101.SIDE_LENGTH),
        spatial_transforms.CenterCrop(UCF101.SIDE_LENGTH),
        spatial_transforms.ToTensor(norm_value=norm_value)
    ]
    if normalize:
        transforms.append(normalization_transform)
    spatial_transform = spatial_transforms.Compose(transforms)
    if whole_videos:
        temporal_transform = None
    else:
        temporal_transform = temporal_transforms.LoopPadding(16)
    target_transform = target_transforms.ClassLabel()
    opt = Bunch(
        dataset='ucf101',
        video_path=data_dir,
        annotation_path=annotation_path,
        n_val_samples=1 if whole_videos else 3,
        sample_duration=16
    )
    validation_data = resnets_3d.dataset.get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True)
    return val_loader, validation_data.class_names


class UCF101:
    MEAN = (110.63666788, 103.16065604, 96.29023126)
    NUM_CLASSES = 101
    SIDE_LENGTH = 112

    @staticmethod
    def get_classifier():
        classifier, _ = generate_model(Bunch(
            model='resnext',
            model_depth=101,
            resnet_shortcut='B',
            resnext_cardinality=32,
            n_classes=101,
            sample_size=UCF101.SIDE_LENGTH,
            sample_duration=16,
            no_cuda=not torch.cuda.is_available(),
            arch='resnext-101',
            n_finetune_classes=101,
            ft_begin_index=0,
            pretrain_path=None
        ))

        resume_path = os.environ['UCF101_MODEL']
        checkpoint = torch.load(resume_path)
        classifier.load_state_dict(checkpoint['state_dict'])

        return classifier

    @staticmethod
    def get_data_loaders(batch_size, num_workers):
        data_dir = os.environ['UCF101_DATA_DIR']
        annotation_path = os.environ['UCF101_ANNOTATION_PATH']
        train_loader, _ = get_ucf101_train_data_loader(batch_size=batch_size, num_workers=num_workers,
                                                       data_dir=data_dir, annotation_path=annotation_path)
        val_loader, _ = get_ucf101_val_data_loader(batch_size=batch_size, num_workers=num_workers,
                                                   data_dir=data_dir, annotation_path=annotation_path)
        return train_loader, val_loader

    @staticmethod
    def get_framing(width, keep_size=False):
        return Framing(width=width, image_side_length=UCF101.SIDE_LENGTH, normalize=UCF101.normalize_not_in_place,
                       scale=255., keep_size=keep_size)

    @staticmethod
    def normalize_not_in_place(input):
        params_shape = list(input.size())
        for i in range(1, len(params_shape)):
            params_shape[i] = 1
        mean = input.new_tensor(UCF101.MEAN).view(params_shape)
        return input - mean
