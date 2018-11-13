import argparse

import matplotlib.pyplot as plt
import torch

from imagenet import ImageNet, load_pretrained_imagenet_framing
from resnet import resnet50

GRID_SIZE = 5
BATCH_SIZE = GRID_SIZE * GRID_SIZE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw examples of attacked ImageNet examples')
    parser.add_argument('--framing', required=True, help='Path to pretrained framing')
    parser.add_argument('--output', '-o', default='examples.png', help='Output file')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, data_loader = ImageNet.get_data_loaders(batch_size=BATCH_SIZE, num_workers=0, normalize=False, shuffle_val=True)

    framing = load_pretrained_imagenet_framing(args.framing).to(device)
    classifier = resnet50(pretrained=True).to(device)
    classifier.eval()

    input, target = next(iter(data_loader))
    input = input.to(device)

    with torch.no_grad():
        input_att, _ = framing(input, normalize=False)

    normalized_input = input.clone()
    normalized_input_att = input_att.clone()
    for id in range(BATCH_SIZE):
        ImageNet.normalize(normalized_input[id])
        ImageNet.normalize(normalized_input_att[id])

    with torch.no_grad():
        output = torch.argmax(classifier(normalized_input), dim=1)
        output_att = torch.argmax(classifier(normalized_input_att), dim=1)

    input_att = input_att.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(7.0, 8.0))
    fig.subplots_adjust(wspace=0.05, hspace=0.6)
    for ax in axes.flatten():
        ax.patch.set_visible(False)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            id = x * GRID_SIZE + y
            target_class = ImageNet.ID_TO_CLASS[target[id].item()].replace('_', ' ')
            unatt_class = ImageNet.ID_TO_CLASS[output[id].item()].replace('_', ' ')
            att_class = ImageNet.ID_TO_CLASS[output_att[id].item()].replace('_', ' ')
            axes[x][y].imshow(input_att[id])
            caption = 'correct: {}\nunattacked: {}\nattacked: {}'.format(target_class, unatt_class, att_class)
            axes[x][y].text(0.5, -0.25, caption, horizontalalignment='center', verticalalignment='center',
                            transform=axes[x][y].transAxes, fontsize=4.5)

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close('all')
    print('Examples saved to {}.'.format(args.output))
