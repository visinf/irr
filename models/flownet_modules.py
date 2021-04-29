from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging


def conv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def deconv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def resize2D(inputs, size_targets, mode="bilinear"):
    size_inputs = [inputs.size(2), inputs.size(3)]

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = tf.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = tf.interpolate(inputs, size=size_targets, mode=mode, align_corners=True)

    return resized

def resize2D_as(inputs, output_as, mode="bilinear"):
    size_targets = [output_as.size(2), output_as.size(3)]
    return resize2D(inputs, size_targets, mode=mode)


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    tensor_list = [resize2D_as(x, tensor_as, mode=mode) for x in tensor_list]
    return torch.cat(tensor_list, dim=dim)


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass

        elif "models" in str(type(layer)) and "FlowNet" in str(type(layer)):            
            pass


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).cuda()
    return grids_cuda


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / width_im / div_flow
        flo_h = flow[:, 1] * 2 / height_im / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid, align_corners=True)

        return x_warp