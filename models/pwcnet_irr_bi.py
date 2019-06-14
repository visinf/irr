from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import conv, rescale_flow, upsample2d_as, initialize_msra
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense
from .correlation_package.correlation import Correlation

class PWCNet(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(PWCNet, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # concat and estimate flow
            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b

            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

            flows.append([flow_f, flow_b])

            # upsampling or post-processing
            if l == self.output_level:
                break

        output_dict['flow'] = flows

        if self.training:
            return output_dict
        else:
            output_dict_eval = {}
            output_dict_eval['flow'] = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)                
            return output_dict_eval
