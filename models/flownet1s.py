from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from .flownet_modules import conv, deconv
from .flownet_modules import concatenate_as, upsample2d_as
from .flownet_modules import initialize_msra


class FlowNetS(nn.Module):
    def __init__(self, args):
        super(FlowNetS, self).__init__()

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

        self._conv1   = make_conv(   6,   64, kernel_size=7, stride=2)
        self._conv2   = make_conv(  64,  128, kernel_size=5, stride=2)
        self._conv3   = make_conv( 128,  256, kernel_size=5, stride=2)
        self._conv3_1 = make_conv( 256,  256, kernel_size=3, stride=1)
        self._conv4   = make_conv( 256,  512, kernel_size=3, stride=2)
        self._conv4_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv5   = make_conv( 512,  512, kernel_size=3, stride=2)
        self._conv5_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv6   = make_conv( 512, 1024, kernel_size=3, stride=2)
        self._conv6_1 = make_conv(1024, 1024, kernel_size=3, stride=1)

        def make_deconv(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=True, bias=False)

        self._deconv5 = make_deconv(1024    , 512)
        self._deconv4 = make_deconv(1024 + 2, 256)
        self._deconv3 = make_deconv( 768 + 2, 128)
        self._deconv2 = make_deconv( 384 + 2,  64)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True)

        self._predict_flow6 = make_predict(1024    , 2)
        self._predict_flow5 = make_predict(1024 + 2, 2)
        self._predict_flow4 = make_predict( 768 + 2, 2)
        self._predict_flow3 = make_predict( 384 + 2, 2)
        self._predict_flow2 = make_predict( 192 + 2, 2)

        def make_upsample(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=False, bias=False)

        self._upsample_flow6_to_5 = make_upsample(2, 2)
        self._upsample_flow5_to_4 = make_upsample(2, 2)
        self._upsample_flow4_to_3 = make_upsample(2, 2)
        self._upsample_flow3_to_2 = make_upsample(2, 2)

        initialize_msra(self.modules())

    def forward(self, inputs):
        conv1 = self._conv1(inputs)
        conv2 = self._conv2(conv1)
        conv3_1 = self._conv3_1(self._conv3(conv2))
        conv4_1 = self._conv4_1(self._conv4(conv3_1))
        conv5_1 = self._conv5_1(self._conv5(conv4_1))
        conv6_1 = self._conv6_1(self._conv6(conv5_1))

        predict_flow6        = self._predict_flow6(conv6_1)

        upsampled_flow6_to_5 = self._upsample_flow6_to_5(predict_flow6)
        deconv5              = self._deconv5(conv6_1)
        concat5              = concatenate_as((conv5_1, deconv5, upsampled_flow6_to_5), conv5_1, dim=1)
        predict_flow5        = self._predict_flow5(concat5)

        upsampled_flow5_to_4 = self._upsample_flow5_to_4(predict_flow5)
        deconv4              = self._deconv4(concat5)
        concat4              = concatenate_as((conv4_1, deconv4, upsampled_flow5_to_4), conv4_1, dim=1)
        predict_flow4        = self._predict_flow4(concat4)

        upsampled_flow4_to_3 = self._upsample_flow4_to_3(predict_flow4)
        deconv3              = self._deconv3(concat4)
        concat3              = concatenate_as((conv3_1, deconv3, upsampled_flow4_to_3), conv3_1, dim=1)
        predict_flow3        = self._predict_flow3(concat3)

        upsampled_flow3_to_2 = self._upsample_flow3_to_2(predict_flow3)
        deconv2              = self._deconv2(concat3)
        concat2              = concatenate_as((conv2, deconv2, upsampled_flow3_to_2), conv2, dim=1)
        predict_flow2        = self._predict_flow2(concat2)

        if self.training:
            return predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6
        else:
            return predict_flow2


class FlowNet1S(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(FlowNet1S, self).__init__()
        self._flownets = FlowNetS(args)
        self._div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self._flownets(inputs)
            output_dict['flow2'] = flow2
            output_dict['flow3'] = flow3
            output_dict['flow4'] = flow4
            output_dict['flow5'] = flow5
            output_dict['flow6'] = flow6
        else:
            flow2 = self._flownets(inputs)
            output_dict['flow1'] = (1.0 / self._div_flow) * upsample2d_as(flow2, im1, mode="bilinear")

        return output_dict
