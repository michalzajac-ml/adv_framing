import argparse
import os

from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Adversarial Framing')
    parser.add_argument('--dataset', required=True, choices=['imagenet', 'ucf101'])
    parser.add_argument('--width', '-w', type=int, required=True, help='Width of the framing')
    parser.add_argument('--keep-size', action='store_true',
                        help='If set, image will be rescaled before applying framing, so that unattacked and attacked '
                             'images have the same size.')
    parser.add_argument('--target', type=int,
                        help='Target class. If unspecified, untargeted attack will be performed. If set to -1, '
                             'target will be chosen randomly. Note that in targeted attack we aim for higher accuracy '
                             'while in untargeted attack we aim for lower accuracy.')
    parser.add_argument('--run-id', help='Identifier of this run')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=1.0, type=float, help='Initial learning rate for the framing')
    parser.add_argument('--lr-decay-wait', default=1, type=int, help='How often (in epochs) is lr being decayed')
    parser.add_argument('--lr-decay-coefficient', default=1.0, type=float,
                        help='When learning rate is being decayed, it is multiplied by this number')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to store checkpoints')
    parser.add_argument('--log-dir', default='logs', help='Directory to store log file')
    parser.add_argument('--tb-dir', default='tensorboard', help='Directory to store TensorBoard logs')
    parser.add_argument('--print-freq', default=20, type=int, help='Print frequency (in batches)')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loading workers for, each of train and eval data will get that many')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    Trainer(args).train(args.epochs)
