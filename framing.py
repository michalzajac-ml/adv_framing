import torch
import torch.nn as nn
import torch.nn.functional as F


class Framing(nn.Module):
    def __init__(self, width, image_side_length, normalize, scale, keep_size=False):
        super(Framing, self).__init__()

        self.width = width
        # If keep_size is True, unattacked and attacked images will have the same size.
        self.keep_size = keep_size
        if keep_size:
            self.length = image_side_length - width
        else:
            self.length = image_side_length + width
        self.attack_shape = [3, self.width, self.length, 4]

        self.attack = nn.Parameter(torch.empty(self.attack_shape).normal_(0, 1))
        self.sigmoid = nn.Sigmoid()
        self.normalize = normalize
        self.scale = scale

    def forward(self, input, normalize=True):
        attack = self.scale * self.sigmoid(self.attack)
        if normalize:
            attack = self.normalize(attack)

        input_size = input.size()
        attacked_size = list(input_size)
        if not self.keep_size:
            attacked_size[-2] = self.length + self.width
            attacked_size[-1] = self.length + self.width

        if len(input_size) == 5:
            # swap color and time dimensions, merge batch and time dimensions
            input = input.permute(0, 2, 1, 3, 4).contiguous().view(-1, input_size[1], input_size[3], input_size[4])

        attacked_size_merged = list(input.size())
        if not self.keep_size:
            attacked_size_merged[-2] = self.length + self.width
            attacked_size_merged[-1] = self.length + self.width

        if self.keep_size:
            inner = F.interpolate(input, size=(self.length - self.width, self.length - self.width), mode='bilinear')
        else:
            inner = input

        framed_input = input.new_zeros(attacked_size_merged)
        framed_input[..., self.width:-self.width, self.width:-self.width] = inner
        framed_input[..., :self.width, :-self.width] = attack[..., 0]
        framed_input[..., :-self.width, -self.width:] = attack[..., 1].transpose(1, 2)
        framed_input[..., -self.width:, self.width:] = attack[..., 2]
        framed_input[..., self.width:, :self.width] = attack[..., 3].transpose(1, 2)

        if len(input_size) == 5:
            framed_input = framed_input.view(attacked_size[0], attacked_size[2], attacked_size[1], attacked_size[3],
                                             attacked_size[4])
            framed_input = framed_input.permute(0, 2, 1, 3, 4)

        return framed_input, attack
