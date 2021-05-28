## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from utils.interpolation import Interp2, Interp2MaskBinary
from utils.interpolation import Meshgrid
import numpy as np


def denormalize_coords(xx, yy, width, height):
    """ scale indices from [-1, 1] to [0, width/height] """
    xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
    yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
    return xx, yy


def normalize_coords(xx, yy, width, height):
    """ scale indices from [0, width/height] to [-1, 1] """
    xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
    yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
    return xx, yy


def apply_transform_to_params(theta0, theta_transform):
    a1 = theta0[:, 0]
    a2 = theta0[:, 1]
    a3 = theta0[:, 2]
    a4 = theta0[:, 3]
    a5 = theta0[:, 4]
    a6 = theta0[:, 5]
    #
    b1 = theta_transform[:, 0]
    b2 = theta_transform[:, 1]
    b3 = theta_transform[:, 2]
    b4 = theta_transform[:, 3]
    b5 = theta_transform[:, 4]
    b6 = theta_transform[:, 5]
    #
    c1 = a1 * b1 + a4 * b2
    c2 = a2 * b1 + a5 * b2
    c3 = b3 + a3 * b1 + a6 * b2
    c4 = a1 * b4 + a4 * b5
    c5 = a2 * b4 + a5 * b5
    c6 = b6 + a3 * b4 + a6 * b5
    #
    new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
    return new_theta


class _IdentityParams(nn.Module):
    def __init__(self):
        super(_IdentityParams, self).__init__()
        self._batch_size = 0
        self.register_buffer("_o", torch.FloatTensor())
        self.register_buffer("_i", torch.FloatTensor())

    def _update(self, batch_size):
        torch.zeros([batch_size, 1], out=self._o)
        torch.ones([batch_size, 1], out=self._i)
        return torch.cat([self._i, self._o, self._o, self._o, self._i, self._o], dim=1)

    def forward(self, batch_size):
        if self._batch_size != batch_size:
            self._identity_params = self._update(batch_size)
            self._batch_size = batch_size
        return self._identity_params


class RandomMirror(nn.Module):
    def __init__(self, vertical=True, p=0.5):
        super(RandomMirror, self).__init__()
        self._batch_size = 0
        self._p = p
        self._vertical = vertical
        self.register_buffer("_mirror_probs", torch.FloatTensor())

    def update_probs(self, batch_size):
        torch.ones([batch_size, 1], out=self._mirror_probs)
        self._mirror_probs *= self._p

    def forward(self, theta1, theta2):
        batch_size = theta1.size(0)
        if batch_size != self._batch_size:
            self.update_probs(batch_size)
            self._batch_size = batch_size

        # apply random sign to a1 a2 a3 (these are the guys responsible for x)
        sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
        i = torch.ones_like(sign)
        horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
        theta1 *= horizontal_mirror
        theta2 *= horizontal_mirror

        # apply random sign to a4 a5 a6 (these are the guys responsible for y)
        if self._vertical:
            sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
            vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
            theta1 *= vertical_mirror
            theta2 *= vertical_mirror

        return theta1, theta2


class RandomCrop(nn.Module):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, crop):
        super(RandomCrop, self).__init__()
        self._crop_size = crop
        self.register_buffer("_x", torch.LongTensor())
        self.register_buffer("_y", torch.LongTensor())

    def forward(self, im1, im2, flo):
        batch_size, _, height, width = im1.size()
        crop_height, crop_width = self._crop_size

        # check whether there is anything to do
        if any(self._size < 1):
            return im1, im2, flo

        # get starting positions
        self._x.random_(0, width - crop_width)
        self._y.random_(0, height - crop_height)

        im1 = im1[:, :, self._y:self._y + crop_height, self._x:self._x + crop_width]
        im2 = im2[:, :, self._y:self._y + crop_height, self._x:self._x + crop_width]
        flo = flo[:, :, self._y:self._y + crop_height, self._x:self._x + crop_width]


class RandomAffineFlow(nn.Module):
    def __init__(self, args, addnoise=True):
        super(RandomAffineFlow, self).__init__()
        self._args = args
        self._interp2 = Interp2(clamp=False)
        self._flow_interp2 = Interp2(clamp=False)
        self._meshgrid = Meshgrid()
        self._identity = _IdentityParams()
        self._random_mirror = RandomMirror()
        self._addnoise = addnoise
        self.register_buffer("_noise1", torch.FloatTensor())
        self.register_buffer("_noise2", torch.FloatTensor())
        self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
        self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))

    def inverse_transform_coords(self, width, height, thetas, offset_x=None, offset_y=None):
        xx, yy = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)

        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
        a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
        a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
        a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
        a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
        a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)

        def _unsqueeze12(u):
            return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

        a1 = _unsqueeze12(thetas[:, 0])
        a2 = _unsqueeze12(thetas[:, 1])
        a3 = _unsqueeze12(thetas[:, 2])
        a4 = _unsqueeze12(thetas[:, 3])
        a5 = _unsqueeze12(thetas[:, 4])
        a6 = _unsqueeze12(thetas[:, 5])
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = xx - a3
        yhat = yy - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat

        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        x = self._xbounds
        y = self._ybounds
        #
        a1 = torch.unsqueeze(thetas[:, 0], dim=1)
        a2 = torch.unsqueeze(thetas[:, 1], dim=1)
        a3 = torch.unsqueeze(thetas[:, 2], dim=1)
        a4 = torch.unsqueeze(thetas[:, 3], dim=1)
        a5 = torch.unsqueeze(thetas[:, 4], dim=1)
        a6 = torch.unsqueeze(thetas[:, 5], dim=1)
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        #
        invalid = (
                      (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                  ).sum(dim=1, keepdim=True) > 0

        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new(batch_size, 1).zero_()
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom).byte()

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid.float() * theta_try + (1 - invalid.float()) * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas)

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self._interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2):
        batch_size, channels, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.stack([u, v], dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        # interp2
        transformed = self._flow_interp2(new_flow, xq, yq)
        return transformed

    def forward(self, example_dict):
        im1 = example_dict["input1"]
        im2 = example_dict["input2"]
        flo = example_dict["target1"]

        batch_size = im1.size(0)
        height = im1.size(2)
        width = im1.size(3)

        # identity = no transform
        theta0 = self._identity(batch_size)

        # # global transform
        theta1 = self.apply_random_transforms_to_params(
            theta0,
            max_translate=0.2,
            min_zoom=1.0, max_zoom=1.5,
            min_squeeze=0.86, max_squeeze=1.16,
            min_rotate=-0.2, max_rotate=0.2,
            validate_size=[height, width])

        # relative transform
        theta2 = self.apply_random_transforms_to_params(
            theta1,
            max_translate=0.015,
            min_zoom=0.985, max_zoom=1.015,
            min_squeeze=1.0, max_squeeze=1.0,
            min_rotate=-0.015, max_rotate=0.015,
            validate_size=[height, width])

        # random flip images
        theta1, theta2 = self._random_mirror(theta1, theta2)

        im1 = self.transform_image(im1, theta1)
        im2 = self.transform_image(im2, theta2)
        flo = self.transform_flow(flo, theta1, theta2)

        if self._addnoise:
            stddev = np.random.uniform(0.0, 0.04)
            self._noise1.resize_as_(im1)
            self._noise2.resize_as_(im2)
            self._noise1.normal_(std=stddev)
            self._noise2.normal_(std=stddev)
            im1 += self._noise1
            im2 += self._noise2
            im1.clamp_(0.0, 1.0)
            im2.clamp_(0.0, 1.0)

        # construct updated dictionaries
        example_dict["input1"] = im1
        example_dict["input2"] = im2
        example_dict["target1"] = flo

        return example_dict


class RandomAffineFlowOcc(nn.Module):
    def __init__(self, args, addnoise=True, crop=None):
        super(RandomAffineFlowOcc, self).__init__()
        self._args = args
        self._interp2 = Interp2(clamp=False)
        self._flow_interp2 = Interp2(clamp=False)
        self._meshgrid = Meshgrid()
        self._identity = _IdentityParams()
        self._random_mirror = RandomMirror()
        self._addnoise = addnoise
        self._crop = crop

        self.register_buffer("_noise1", torch.FloatTensor())
        self.register_buffer("_noise2", torch.FloatTensor())
        self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
        self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
        self.register_buffer("_x", torch.IntTensor(1))
        self.register_buffer("_y", torch.IntTensor(1))

    def inverse_transform_coords(self, width, height, thetas, offset_x=None, offset_y=None):
        xx, yy = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)

        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
        a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
        a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
        a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
        a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
        a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)

        def _unsqueeze12(u):
            return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

        a1 = _unsqueeze12(thetas[:, 0])
        a2 = _unsqueeze12(thetas[:, 1])
        a3 = _unsqueeze12(thetas[:, 2])
        a4 = _unsqueeze12(thetas[:, 3])
        a5 = _unsqueeze12(thetas[:, 4])
        a6 = _unsqueeze12(thetas[:, 5])
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = xx - a3
        yhat = yy - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat

        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        x = self._xbounds
        y = self._ybounds
        #
        a1 = torch.unsqueeze(thetas[:, 0], dim=1)
        a2 = torch.unsqueeze(thetas[:, 1], dim=1)
        a3 = torch.unsqueeze(thetas[:, 2], dim=1)
        a4 = torch.unsqueeze(thetas[:, 3], dim=1)
        a5 = torch.unsqueeze(thetas[:, 4], dim=1)
        a6 = torch.unsqueeze(thetas[:, 5], dim=1)
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        #
        invalid = (
                      (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                  ).sum(dim=1, keepdim=True) > 0

        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new(batch_size, 1).zero_()
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom).byte()

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid.float() * theta_try + (1. - invalid.float()) * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas)

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self._interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2):
        batch_size, channels, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.stack([u, v], dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        # interp2
        transformed = self._flow_interp2(new_flow, xq, yq)
        return transformed

    def check_out_of_bound(self, flow, occ, batch_size):
        _, _, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        xx, yy = self._meshgrid(width=width, height=height, device=flow.device, dtype=flow.dtype)
        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()
        xx = xx.expand(batch_size, -1, -1) + u
        yy = yy.expand(batch_size, -1, -1) + v

        out_of_bound = ((xx < 0) | (yy < 0) | (xx >= width) | (yy >= height)).float().unsqueeze(1)
        occ = torch.clamp(out_of_bound + occ, 0, 1)

        return occ

    def random_crop(self, im1, im2, flo_f, flo_b, occ1, occ2):

        _, _, height, width = im1.size()
        crop_height, crop_width = self._crop

        # get starting positions
        self._x.random_(0, width - crop_width + 1)
        self._y.random_(0, height - crop_height + 1)
        str_x = int(self._x)
        str_y = int(self._y)
        end_x = int(self._x + crop_width)
        end_y = int(self._y + crop_height)

        im1 = im1[:, :, str_y:end_y, str_x:end_x]
        im2 = im2[:, :, str_y:end_y, str_x:end_x]
        flo_f = flo_f[:, :, str_y:end_y, str_x:end_x]
        flo_b = flo_b[:, :, str_y:end_y, str_x:end_x]
        occ1 = occ1[:, :, str_y:end_y, str_x:end_x]
        occ2 = occ2[:, :, str_y:end_y, str_x:end_x]

        return im1, im2, flo_f, flo_b, occ1, occ2

    def forward(self, example_dict):
        im1 = example_dict["input1"]
        im2 = example_dict["input2"]
        flo_f = example_dict["target1"]
        flo_b = example_dict["target2"]
        occ1 = example_dict["target_occ1"]
        occ2 = example_dict["target_occ2"]

        batch_size = im1.size(0)
        height = im1.size(2)
        width = im1.size(3)

        # identity = no transform
        theta0 = self._identity(batch_size)

        # # global transform
        theta1 = self.apply_random_transforms_to_params(
            theta0,
            max_translate=0.2,
            min_zoom=1.0, max_zoom=1.5,
            min_squeeze=0.86, max_squeeze=1.16,
            min_rotate=-0.2, max_rotate=0.2,
            validate_size=[height, width])

        # relative transform
        theta2 = self.apply_random_transforms_to_params(
            theta1,
            max_translate=0.015,
            min_zoom=0.985, max_zoom=1.015,
            min_squeeze=1.0, max_squeeze=1.0,
            min_rotate=-0.015, max_rotate=0.015,
            validate_size=[height, width])

        # random flip images
        theta1, theta2 = self._random_mirror(theta1, theta2)

        im1 = self.transform_image(im1, theta1)
        im2 = self.transform_image(im2, theta2)
        flo_f = self.transform_flow(flo_f, theta1, theta2)
        flo_b = self.transform_flow(flo_b, theta2, theta1)
        occ1 = self.transform_image(occ1, theta1)
        occ2 = self.transform_image(occ2, theta2)

        if self._addnoise:
            stddev = np.random.uniform(0.0, 0.04)
            self._noise1.resize_as_(im1)
            self._noise2.resize_as_(im2)
            self._noise1.normal_(std=stddev)
            self._noise2.normal_(std=stddev)
            im1 += self._noise1
            im2 += self._noise2
            im1.clamp_(0.0, 1.0)
            im2.clamp_(0.0, 1.0)

        if self._crop is not None:
            im1, im2, flo_f, flo_b, occ1, occ2 = self.random_crop(im1, im2, flo_f, flo_b, occ1, occ2)

        occ1 = self.check_out_of_bound(flo_f, occ1, batch_size)
        occ2 = self.check_out_of_bound(flo_b, occ2, batch_size)

        example_dict["input1"] = im1
        example_dict["input2"] = im2
        example_dict["target1"] = flo_f
        example_dict["target2"] = flo_b
        example_dict["target_occ1"] = occ1
        example_dict["target_occ2"] = occ2

        return example_dict


class RandomAffineFlowOccSintel(nn.Module):
    def __init__(self, args, addnoise=True, crop=None):
        super(RandomAffineFlowOccSintel, self).__init__()
        self._args = args
        self._interp2 = Interp2(clamp=False)
        self._flow_interp2 = Interp2(clamp=False)
        self._meshgrid = Meshgrid()
        self._identity = _IdentityParams()
        self._random_mirror = RandomMirror()
        self._addnoise = addnoise
        self._crop = crop

        self.register_buffer("_noise1", torch.FloatTensor())
        self.register_buffer("_noise2", torch.FloatTensor())
        self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
        self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
        self.register_buffer("_x", torch.IntTensor(1))
        self.register_buffer("_y", torch.IntTensor(1))

    def inverse_transform_coords(self, width, height, thetas, offset_x=None, offset_y=None):
        xx, yy = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)

        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
        a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
        a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
        a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
        a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
        a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)

        def _unsqueeze12(u):
            return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

        a1 = _unsqueeze12(thetas[:, 0])
        a2 = _unsqueeze12(thetas[:, 1])
        a3 = _unsqueeze12(thetas[:, 2])
        a4 = _unsqueeze12(thetas[:, 3])
        a5 = _unsqueeze12(thetas[:, 4])
        a6 = _unsqueeze12(thetas[:, 5])
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = xx - a3
        yhat = yy - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat

        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        x = self._xbounds
        y = self._ybounds
        #
        a1 = torch.unsqueeze(thetas[:, 0], dim=1)
        a2 = torch.unsqueeze(thetas[:, 1], dim=1)
        a3 = torch.unsqueeze(thetas[:, 2], dim=1)
        a4 = torch.unsqueeze(thetas[:, 3], dim=1)
        a5 = torch.unsqueeze(thetas[:, 4], dim=1)
        a6 = torch.unsqueeze(thetas[:, 5], dim=1)
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        #
        invalid = (
                      (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                  ).sum(dim=1, keepdim=True) > 0

        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new(batch_size, 1).zero_()
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom).byte()

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid.float() * theta_try + (1 - invalid.float()) * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas)

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self._interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2):
        batch_size, channels, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.stack([u, v], dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        # interp2
        transformed = self._flow_interp2(new_flow, xq, yq)
        return transformed

    def check_out_of_bound(self, flow, occ, batch_size):
        _, _, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        xx, yy = self._meshgrid(width=width, height=height, device=flow.device, dtype=flow.dtype)
        xx = torch.unsqueeze(xx, dim=0)
        yy = torch.unsqueeze(yy, dim=0)
        xx = xx.expand(batch_size, -1, -1) + u
        yy = yy.expand(batch_size, -1, -1) + v

        out_of_bound = ((xx < 0) | (yy < 0) | (xx >= width) | (yy >= height)).float().unsqueeze(1)
        occ = torch.clamp(out_of_bound + occ, 0, 1)

        return occ

    def random_crop(self, im1, im2, flo_f, occ1):

        _, _, height, width = im1.size()
        crop_height, crop_width = self._crop

        # get starting positions
        self._x.random_(0, width - crop_width + 1)
        self._y.random_(0, height - crop_height + 1)
        str_x = int(self._x)
        str_y = int(self._y)
        end_x = int(self._x + crop_width)
        end_y = int(self._y + crop_height)

        im1 = im1[:, :, str_y:end_y, str_x:end_x]
        im2 = im2[:, :, str_y:end_y, str_x:end_x]
        flo_f = flo_f[:, :, str_y:end_y, str_x:end_x]
        occ1 = occ1[:, :, str_y:end_y, str_x:end_x]

        return im1, im2, flo_f, occ1

    def forward(self, example_dict):
        im1 = example_dict["input1"]
        im2 = example_dict["input2"]
        flo_f = example_dict["target1"]
        occ1 = example_dict["target_occ1"]

        batch_size = im1.size(0)
        height = im1.size(2)
        width = im1.size(3)

        # identity = no transform
        theta0 = self._identity(batch_size)

        # # global transform
        theta1 = self.apply_random_transforms_to_params(
            theta0,
            max_translate=0.2,
            min_zoom=1.0, max_zoom=1.5,
            min_squeeze=0.86, max_squeeze=1.16,
            min_rotate=-0.2, max_rotate=0.2,
            validate_size=[height, width])

        # relative transform
        theta2 = self.apply_random_transforms_to_params(
            theta1,
            max_translate=0.015,
            min_zoom=0.985, max_zoom=1.015,
            min_squeeze=1.0, max_squeeze=1.0,
            min_rotate=-0.015, max_rotate=0.015,
            validate_size=[height, width])

        # random flip images
        theta1, theta2 = self._random_mirror(theta1, theta2)

        im1 = self.transform_image(im1, theta1)
        im2 = self.transform_image(im2, theta2)
        flo_f = self.transform_flow(flo_f, theta1, theta2)
        occ1 = self.transform_image(occ1, theta1)

        if self._addnoise:
            stddev = np.random.uniform(0.0, 0.04)
            self._noise1.resize_as_(im1)
            self._noise2.resize_as_(im2)
            self._noise1.normal_(std=stddev)
            self._noise2.normal_(std=stddev)
            im1 += self._noise1
            im2 += self._noise2
            im1.clamp_(0.0, 1.0)
            im2.clamp_(0.0, 1.0)

        if self._crop is not None:
            im1, im2, flo_f, occ1 = self.random_crop(im1, im2, flo_f, occ1)

        occ1 = self.check_out_of_bound(flo_f, occ1, batch_size)

        example_dict["input1"] = im1
        example_dict["input2"] = im2
        example_dict["target1"] = flo_f
        example_dict["target_occ1"] = occ1

        return example_dict


class RandomAffineFlowOccKITTI(nn.Module):
    def __init__(self, args, addnoise=True, crop=None):
        super(RandomAffineFlowOccKITTI, self).__init__()
        self._args = args
        self._interp2 = Interp2(clamp=False)
        self._flow_interp2 = Interp2MaskBinary(clamp=False)
        self._meshgrid = Meshgrid()
        self._identity = _IdentityParams()
        self._random_mirror = RandomMirror(vertical=False)
        self._addnoise = addnoise
        self._crop = crop

        self.register_buffer("_noise1", torch.FloatTensor())
        self.register_buffer("_noise2", torch.FloatTensor())
        self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
        self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
        self.register_buffer("_x", torch.IntTensor(1))
        self.register_buffer("_y", torch.IntTensor(1))

    def inverse_transform_coords(self, width, height, thetas, offset_x=None, offset_y=None):
        xx, yy = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)

        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
        a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
        a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
        a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
        a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
        a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self._meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)

        def _unsqueeze12(u):
            return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

        a1 = _unsqueeze12(thetas[:, 0])
        a2 = _unsqueeze12(thetas[:, 1])
        a3 = _unsqueeze12(thetas[:, 2])
        a4 = _unsqueeze12(thetas[:, 3])
        a5 = _unsqueeze12(thetas[:, 4])
        a6 = _unsqueeze12(thetas[:, 5])
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = xx - a3
        yhat = yy - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat

        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        x = self._xbounds
        y = self._ybounds
        #
        a1 = torch.unsqueeze(thetas[:, 0], dim=1)
        a2 = torch.unsqueeze(thetas[:, 1], dim=1)
        a3 = torch.unsqueeze(thetas[:, 2], dim=1)
        a4 = torch.unsqueeze(thetas[:, 3], dim=1)
        a5 = torch.unsqueeze(thetas[:, 4], dim=1)
        a6 = torch.unsqueeze(thetas[:, 5], dim=1)
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        #
        invalid = (
                      (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                  ).sum(dim=1, keepdim=True) > 0

        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new(batch_size, 1).zero_()
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom).byte()

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid.float() * theta_try + (1 - invalid.float()) * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas)

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self._interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2, valid_mask):
        batch_size, channels, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.stack([u, v], dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        # interp2
        # transformed = self._interp2(new_flow, xq, yq)
        transformed, valid_mask = self._flow_interp2(new_flow, xq, yq, valid_mask)
        return transformed, valid_mask

    def check_out_of_bound(self, flow, occ, batch_size):
        _, _, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        xx, yy = self._meshgrid(width=width, height=height, device=flow.device, dtype=flow.dtype)
        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()
        xx = xx.expand(batch_size, -1, -1) + u
        yy = yy.expand(batch_size, -1, -1) + v

        out_of_bound = ((xx < 0) | (yy < 0) | (xx >= width) | (yy >= height)).float().unsqueeze(1)
        occ = torch.clamp(out_of_bound + occ, 0, 1)

        return occ

    def random_crop(self, im1, im2, flo_f, valid_mask):

        _, _, height, width = im1.size()
        crop_height, crop_width = self._crop

        # get starting positions
        self._x.random_(0, width - crop_width + 1)
        self._y.random_(0, height - crop_height + 1)
        str_x = int(self._x)
        str_y = int(self._y)
        end_x = int(self._x + crop_width)
        end_y = int(self._y + crop_height)

        im1 = im1[:, :, str_y:end_y, str_x:end_x]
        im2 = im2[:, :, str_y:end_y, str_x:end_x]
        flo_f = flo_f[:, :, str_y:end_y, str_x:end_x]
        valid_mask = valid_mask[:, :, str_y:end_y, str_x:end_x]

        return im1, im2, flo_f, valid_mask

    def forward(self, example_dict):
        im1 = example_dict["input1"]
        im2 = example_dict["input2"]
        flo_f = example_dict["target1"]
        valid_mask = example_dict["input_valid"]

        batch_size = im1.size(0)
        height = im1.size(2)
        width = im1.size(3)

        # identity = no transform
        theta0 = self._identity(batch_size)

        # # global transform
        theta1 = self.apply_random_transforms_to_params(
            theta0,
            max_translate=0.04,
            min_zoom=0.98, max_zoom=1.02,
            min_squeeze=1.0, max_squeeze=1.0,
            min_rotate=-0.01, max_rotate=0.01,
            validate_size=[height, width])

        # relative transform
        theta2 = self.apply_random_transforms_to_params(
            theta1,
            max_translate=0.005,
            min_zoom=0.99, max_zoom=1.01,
            min_squeeze=1.0, max_squeeze=1.0,
            min_rotate=-0.01, max_rotate=0.01,
            validate_size=[height, width])

        # random flip images
        theta1, theta2 = self._random_mirror(theta1, theta2)

        im1 = self.transform_image(im1, theta1)
        im2 = self.transform_image(im2, theta2)
        flo_f, valid_mask = self.transform_flow(flo_f, theta1, theta2, valid_mask)


        if self._addnoise:
            stddev = np.random.uniform(0.0, 0.04)
            self._noise1.resize_as_(im1)
            self._noise2.resize_as_(im2)
            self._noise1.normal_(std=stddev)
            self._noise2.normal_(std=stddev)
            im1 += self._noise1
            im2 += self._noise2
            im1.clamp_(0.0, 1.0)
            im2.clamp_(0.0, 1.0)

        if self._crop is not None:
            im1, im2, flo_f, valid_mask = self.random_crop(im1, im2, flo_f, valid_mask)

        example_dict["input1"] = im1
        example_dict["input2"] = im2
        example_dict["target1"] = flo_f
        example_dict["input_valid"] = valid_mask

        return example_dict
