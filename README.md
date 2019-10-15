# Adversarial Framing


This is the official implementation of the experiments from the paper
"Adversarial Framing for Image and Video Classification" ([video](https://youtu.be/PrU9R6eFNTs))
by Michał Zając, Konrad Żołna, Negar Rostamzadeh and Pedro Pinheiro.

The [code](https://github.com/kenshohara/3D-ResNets-PyTorch) from the paper "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?"
is also included in `deps/resnets_3d` folder, as we attack the model from that paper.

Our code was originally forked from [Classifier-agnostic saliency map extraction](https://github.com/kondiz/casme) repository.

## Requirements

The code uses Python 3 and packages listed in `requirements.txt`. If you use pip, you can install them by `pip install -r requirements.txt`.

## Datasets preparation

### Imagenet
1. Follow the [instructions](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)
to download and unpack the dataset.
2. Set environment variable `IMAGENET_DATA_DIR` with the directory to the dataset by `export IMAGENET_DATA_DIR=/your/imagenet/dir`,
where `/your/imagenet/dir` should contain `train` and `val` folders (as in the instructions above).

### UCF101
1. Follow the [instructions](https://github.com/kenshohara/3D-ResNets-PyTorch#ucf-101)
to download, unpack and preprocess the dataset.
2. Set data environment variables `UCF101_DATA_DIR` and `UCF101_ANNOTATION_PATH`.
  - `export UCF101_DATA_DIR=/your/data/dir`, where `/your/data/dir` is `jpg_video_directory` from the instruction above.
  - `export UCF101_ANNOTATION_PATH=/your/annotation/path`, where `/your/annotation/path` is a path to the file `ucf101_01.json` created with the above instruction.
3. Download a pretrained model called `resnext-101-kinetics-ucf101_split1.pth` from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M).
The model comes from the paper [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://github.com/kenshohara/3D-ResNets-PyTorch)
4. Set model environment variable `UCF101_MODEL` by `export UCF101_MODEL=/your/model/path`.

## Running
- First run `export PYTHONPATH=$PYTHONPATH:deps` from the main project directory.
- To reproduce untargeted ImageNet experiments, run `python3 main.py --dataset imagenet --width $WIDTH --epochs 5 --lr 0.1 --lr-decay-wait 2 --lr-decay-coefficient 0.1`,
where you should set `WIDTH` of the framing.
- To reproduce untargeted UCF101 experiments, run `python3 main.py --dataset ucf101 --width $WIDTH --epochs 60 --lr 0.03 --lr-decay-wait 15 --lr-decay-coefficient 0.3`,
where you should set `WIDTH` of the framing.
- To draw some examples of attacks on ImageNet, run `python3 draw_examples_imagenet.py --framing $CHECKPOINT`. As a `CHECKPOINT` you can use some model from `pretrained` directory.

## Citation
If you found this code useful, please use the following citation:

    @inproceedings{zajac2019adversarial,
      title={Adversarial framing for image and video classification},
      author={Zajac, Micha{\l} and Zo{\l}na, Konrad and Rostamzadeh, Negar and Pinheiro, Pedro O},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={33},
      pages={10077--10078},
      year={2019}
    }
