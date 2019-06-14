from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from .flownet_modules import conv, deconv
from .flownet_modules import concatenate_as, upsample2d_as
from .flownet_modules import initialize_msra
from .flownet_modules import WarpingLayer
from .irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc

class FlowNetS(nn.Module):
    def __init__(self, args):
        super(FlowNetS, self).__init__()

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

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

        self._deconv_occ5 = make_deconv(1024    , 512)
        self._deconv_occ4 = make_deconv(1024 + 1, 256)
        self._deconv_occ3 = make_deconv( 768 + 1, 128)
        self._deconv_occ2 = make_deconv( 384 + 1,  64)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True)

        self._predict_flow6 = make_predict(1024    , 2)
        self._predict_flow5 = make_predict(1024 + 2, 2)
        self._predict_flow4 = make_predict( 768 + 2, 2)
        self._predict_flow3 = make_predict( 384 + 2, 2)
        self._predict_flow2 = make_predict( 128 + 2, 2)

        self._predict_occ6 = make_predict(1024    , 1)
        self._predict_occ5 = make_predict(1024 + 1, 1)
        self._predict_occ4 = make_predict( 768 + 1, 1)
        self._predict_occ3 = make_predict( 384 + 1, 1)
        self._predict_occ2 = make_predict( 128 + 1, 1)

        def make_upsample(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=False, bias=False)

        self._upsample_flow6_to_5 = make_upsample(2, 2)
        self._upsample_flow5_to_4 = make_upsample(2, 2)
        self._upsample_flow4_to_3 = make_upsample(2, 2)
        self._upsample_flow3_to_2 = make_upsample(2, 2)

        self._upsample_occ6_to_5 = make_upsample(1, 1)
        self._upsample_occ5_to_4 = make_upsample(1, 1)
        self._upsample_occ4_to_3 = make_upsample(1, 1)
        self._upsample_occ3_to_2 = make_upsample(1, 1)

    def forward(self, conv2_im1, conv3_im1, conv3_im2):

        conv_concat3 = torch.cat((conv3_im1, conv3_im2), dim=1)

        conv3_1 = self._conv3_1(conv_concat3)
        conv4_1 = self._conv4_1(self._conv4(conv3_1))
        conv5_1 = self._conv5_1(self._conv5(conv4_1))
        conv6_1 = self._conv6_1(self._conv6(conv5_1))

        # Flow Decoder
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
        concat2              = concatenate_as((conv2_im1, deconv2, upsampled_flow3_to_2), conv2_im1, dim=1)
        predict_flow2        = self._predict_flow2(concat2)

        # Occ Decoder
        predict_occ6 = self._predict_occ6(conv6_1)

        upsampled_occ6_to_5 = self._upsample_occ6_to_5(predict_occ6)
        deconv_occ5         = self._deconv_occ5(conv6_1)
        concat_occ5         = concatenate_as((conv5_1, deconv_occ5, upsampled_occ6_to_5), conv5_1, dim=1)
        predict_occ5        = self._predict_occ5(concat_occ5)

        upsampled_occ5_to_4 = self._upsample_occ5_to_4(predict_occ5)
        deconv_occ4         = self._deconv_occ4(concat_occ5)
        concat_occ4         = concatenate_as((conv4_1, deconv_occ4, upsampled_occ5_to_4), conv4_1, dim=1)
        predict_occ4        = self._predict_occ4(concat_occ4)

        upsampled_occ4_to_3 = self._upsample_occ4_to_3(predict_occ4)
        deconv_occ3         = self._deconv_occ3(concat_occ4)
        concat_occ3         = concatenate_as((conv3_1, deconv_occ3, upsampled_occ4_to_3), conv3_1, dim=1)
        predict_occ3        = self._predict_occ3(concat_occ3)

        upsampled_occ3_to_2 = self._upsample_occ3_to_2(predict_occ3)
        deconv_occ2         = self._deconv_occ2(concat_occ3)
        concat_occ2         = concatenate_as((conv2_im1, deconv_occ2, upsampled_occ3_to_2), conv2_im1, dim=1)
        predict_occ2        = self._predict_occ2(concat_occ2)

        return predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6, predict_occ2, predict_occ3, predict_occ4, predict_occ5, predict_occ6


class FlowNet1S(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(FlowNet1S, self).__init__()
        self._flownets = FlowNetS(args)
        self._warping_layer = WarpingLayer()
        self._div_flow = div_flow
        self._num_iters = args.num_iters

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

        self._conv1   = make_conv(   3,   32, kernel_size=7, stride=2)
        self._conv2   = make_conv(  32,   64, kernel_size=5, stride=2)
        self._conv3   = make_conv(  64,  128, kernel_size=5, stride=2)

        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)
        self.refine_flow = RefineFlow(2 + 1 + 64)
        self.refine_occ = RefineOcc(1 + 64 + 64)

        initialize_msra(self.modules())

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']

        conv1_im1 = self._conv1(im1)
        conv2_im1 = self._conv2(conv1_im1)
        conv3_im1 = self._conv3(conv2_im1)
        conv3_im1_wp = conv3_im1

        conv1_im2 = self._conv1(im2)
        conv2_im2 = self._conv2(conv1_im2)
        conv3_im2 = self._conv3(conv2_im2)
        conv3_im2_wp = conv3_im2

        out_dict = {}
        out_dict['flow'] = []
        out_dict['flow1'] = []
        out_dict['flow2'] = []
        out_dict['flow3'] = []
        out_dict['flow4'] = []
        out_dict['flow5'] = []
        out_dict['flow6'] = []
        out_dict['occ'] = []
        out_dict['occ1'] = []
        out_dict['occ2'] = []
        out_dict['occ3'] = []
        out_dict['occ4'] = []
        out_dict['occ5'] = []
        out_dict['occ6'] = []

        # warping:
        _, _, height_im, width_im = im1.size()

        # for iterative
        for ii in range(0, self._num_iters):
            flo2_f, flo3_f, flo4_f, flo5_f, flo6_f, occ2_f, occ3_f, occ4_f, occ5_f, occ6_f = self._flownets(conv2_im1,
                                                                                                            conv3_im1,
                                                                                                            conv3_im2_wp)
            flo2_b, flo3_b, flo4_b, flo5_b, flo6_b, occ2_b, occ3_b, occ4_b, occ5_b, occ6_b = self._flownets(conv2_im2,
                                                                                                            conv3_im2,
                                                                                                            conv3_im1_wp)

            if ii == 0:
                out_dict['flow2'].append([flo2_f, flo2_b])
                out_dict['flow3'].append([flo3_f, flo3_b])
                out_dict['flow4'].append([flo4_f, flo4_b])
                out_dict['flow5'].append([flo5_f, flo5_b])
                out_dict['flow6'].append([flo6_f, flo6_b])
                out_dict['occ2'].append([occ2_f, occ2_b])
                out_dict['occ3'].append([occ3_f, occ3_b])
                out_dict['occ4'].append([occ4_f, occ4_b])
                out_dict['occ5'].append([occ5_f, occ5_b])
                out_dict['occ6'].append([occ6_f, occ6_b])
                flo2_f_out = flo2_f
                flo2_b_out = flo2_b
                occ2_f_out = occ2_f
                occ2_b_out = occ2_b
            else:
                out_dict['flow2'].append([flo2_f + out_dict['flow2'][ii - 1][0], flo2_b + out_dict['flow2'][ii - 1][1]])
                out_dict['flow3'].append([flo3_f + out_dict['flow3'][ii - 1][0], flo3_b + out_dict['flow3'][ii - 1][1]])
                out_dict['flow4'].append([flo4_f + out_dict['flow4'][ii - 1][0], flo4_b + out_dict['flow4'][ii - 1][1]])
                out_dict['flow5'].append([flo5_f + out_dict['flow5'][ii - 1][0], flo5_b + out_dict['flow5'][ii - 1][1]])
                out_dict['flow6'].append([flo6_f + out_dict['flow6'][ii - 1][0], flo6_b + out_dict['flow6'][ii - 1][1]])
                out_dict['occ2'].append([occ2_f + out_dict['occ2'][ii - 1][0], occ2_b + out_dict['occ2'][ii - 1][1]])
                out_dict['occ3'].append([occ3_f + out_dict['occ3'][ii - 1][0], occ3_b + out_dict['occ3'][ii - 1][1]])
                out_dict['occ4'].append([occ4_f + out_dict['occ4'][ii - 1][0], occ4_b + out_dict['occ4'][ii - 1][1]])
                out_dict['occ5'].append([occ5_f + out_dict['occ5'][ii - 1][0], occ5_b + out_dict['occ5'][ii - 1][1]])
                out_dict['occ6'].append([occ6_f + out_dict['occ6'][ii - 1][0], occ6_b + out_dict['occ6'][ii - 1][1]])
                flo2_f_out = flo2_f + upsample2d_as(out_dict['flow1'][ii - 1][0], flo2_f, mode="bilinear")
                flo2_b_out = flo2_b + upsample2d_as(out_dict['flow1'][ii - 1][1], flo2_b, mode="bilinear")
                occ2_f_out = occ2_f + upsample2d_as(out_dict['occ1'][ii - 1][0], occ2_f, mode="bilinear")
                occ2_b_out = occ2_b + upsample2d_as(out_dict['occ1'][ii - 1][1], occ2_b, mode="bilinear")

            ## refine layer
            flo2_f_out = upsample2d_as(flo2_f_out, conv2_im1, mode="bilinear")  
            flo2_b_out = upsample2d_as(flo2_b_out, conv2_im2, mode="bilinear") 
            occ2_f_out = upsample2d_as(occ2_f_out, conv2_im1, mode="bilinear")
            occ2_b_out = upsample2d_as(occ2_b_out, conv2_im2, mode="bilinear")

            img1_resize = upsample2d_as(im1, flo2_f_out, mode="bilinear")
            img2_resize = upsample2d_as(im2, flo2_b_out, mode="bilinear")
            img2_warp = self._warping_layer(img2_resize, flo2_f_out, height_im, width_im, self._div_flow)
            img1_warp = self._warping_layer(img1_resize, flo2_b_out, height_im, width_im, self._div_flow)

            # flow refine
            flow_f = self.refine_flow(flo2_f_out.detach(), img1_resize - img2_warp, conv2_im1)
            flow_b = self.refine_flow(flo2_b_out.detach(), img2_resize - img1_warp, conv2_im2)

            # occ refine
            conv2_im2_warp = self._warping_layer(conv2_im2, flow_f, height_im, width_im, self._div_flow)
            conv2_im1_warp = self._warping_layer(conv2_im1, flow_b, height_im, width_im, self._div_flow)

            occ_f = self.refine_occ(occ2_f_out.detach(), conv2_im1, conv2_im1 - conv2_im2_warp)
            occ_b = self.refine_occ(occ2_b_out.detach(), conv2_im2, conv2_im2 - conv2_im1_warp)
            out_dict['flow1'].append([flow_f, flow_b])
            out_dict['occ1'].append([occ_f, occ_b])

            ## upsample layer
            flow_f = upsample2d_as(flow_f, im1, mode="bilinear")
            flow_b = upsample2d_as(flow_b, im2, mode="bilinear")
            out_dict['flow'].append([flow_f, flow_b])

            im2_warp = self._warping_layer(im2, flow_f, height_im, width_im, self._div_flow)
            im1_warp = self._warping_layer(im1, flow_b, height_im, width_im, self._div_flow)
            flow_b_warp = self._warping_layer(flow_b, flow_f, height_im, width_im, self._div_flow)
            flow_f_warp = self._warping_layer(flow_f, flow_b, height_im, width_im, self._div_flow)

            occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([im1, im2_warp, flow_f, flow_b_warp], dim=1))
            occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([im2, im1_warp, flow_b, flow_f_warp], dim=1))

            out_dict['occ'].append([occ_f, occ_b])

            if ii < (self._num_iters - 1):
                flow_f_resized = upsample2d_as(flow_f, conv3_im2, mode="bilinear")
                flow_b_resized = upsample2d_as(flow_b, conv3_im1, mode="bilinear")
                conv3_im2_wp = self._warping_layer(conv3_im2, flow_f_resized, height_im, width_im, self._div_flow)
                conv3_im1_wp = self._warping_layer(conv3_im1, flow_b_resized, height_im, width_im, self._div_flow)
                
        if self.training:
            return out_dict
        else:
            out_dict_eval = {}
            out_dict_eval['flow'] = upsample2d_as(out_dict['flow'][self._num_iters - 1][0], im1, mode="bilinear") / self._div_flow
            out_dict_eval['occ'] = upsample2d_as(out_dict['occ'][self._num_iters - 1][0], im1, mode="bilinear")
            return out_dict_eval
