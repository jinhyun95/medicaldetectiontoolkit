#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.functional as F
import torch


class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling
        self.dim = conv.dim

        if operate_stride1:
            self.C0 = nn.Sequential(conv(cf.n_channels, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
                                    conv(start_filts, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu))

            self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        else:
            self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(
                start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 8, 2, 2)))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=cf.norm, relu=cf.relu))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        self.out_channels = cf.end_filts
        self.P5_conv1 = conv(start_filts*32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list



class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class FPN_DARTS(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN_DARTS, self).__init__()

        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling
        self.dim = conv.dim

        ############ ResNet BackBone ############
        self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)
        ############ ResNet BackBone ############

        ############ Feature Pyramid ############
        self.out_channels = cf.end_filts
        self.temperature = 1.
        self.annealing_rate = 1e-4
        self.fix_architecture = False

        self.C5_Q5 = nn.ModuleList(
            [conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C4_Q4 = nn.ModuleList(
            [conv(start_filts * 16, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 16, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 16, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C3_Q3 = nn.ModuleList(
            [conv(start_filts * 8, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 8, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 8, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C2_Q2 = nn.ModuleList(
            [conv(start_filts * 4, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 4, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 4, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.C5_P5 = nn.ModuleList(
            [conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C4_P4 = nn.ModuleList(
            [conv(start_filts * 16, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 16, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 16, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C3_P3 = nn.ModuleList(
            [conv(start_filts * 8, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 8, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 8, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C2_P2 = nn.ModuleList(
            [conv(start_filts * 4, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 4, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 4, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.C5_Q4 = nn.ModuleList(
            [conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C4_Q3 = nn.ModuleList(
            [conv(start_filts * 16, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 16, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 16, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C3_Q2 = nn.ModuleList(
            [conv(start_filts * 8, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 8, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 8, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.C5_P4 = nn.ModuleList(
            [conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C4_P3 = nn.ModuleList(
            [conv(start_filts * 16, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 16, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 16, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.C3_P2 = nn.ModuleList(
            [conv(start_filts * 8, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(start_filts * 8, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(start_filts * 8, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.Q5_P5 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q4_P4 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q3_P3 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q2_P2 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.Q5_Q4 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q4_Q3 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q3_Q2 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.Q5_P4 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q4_P3 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])
        self.Q3_P2 = nn.ModuleList(
            [conv(self.out_channels, self.out_channels, ks=1, stride=1, pad=0, relu=None),
             conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None),
             conv(self.out_channels, self.out_channels, ks=5, stride=1, pad=2, relu=None)])

        self.Q5_conn = nn.Parameter(torch.rand(1, 1, 1, 1, 3), requires_grad=True)
        self.P5_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 8), requires_grad=True)
        self.P5_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 8), requires_grad=True)
        self.Q4_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.Q4_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.Q3_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.Q3_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.Q2_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.Q2_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 11), requires_grad=True)
        self.P4_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        self.P4_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        self.P3_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        self.P3_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        self.P2_conn_1 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        self.P2_conn_2 = nn.Parameter(torch.rand(1, 1, 1, 1, 15), requires_grad=True)
        ############ Feature Pyramid ############

    def darts(self, conn):
        if self.fix_architecture:
            arg = torch.argmax(torch.squeeze(conn), dim=0).cpu().numpy()[0, 0, 0, 0]
            connected = torch.zeros_like(conn)
            connected[0, 0, 0, 0, arg] = 1.
            return connected
        else:
            return F.softmax(conn / self.temperature, dim=-1)

    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        ############ ResNet Backbone ############
        c0_out = x
        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        ############ ResNet Backbone ############

        ############ Feature Pyramid ############
        # TODO: softmax temperature annealing
        q5_out = []
        q5_out.extend([net(c5_out) for net in self.C5_Q5])
        q5_out = torch.sum(torch.stack(q5_out, dim=-1) * self.darts(self.Q5_conn), dim=-1, keepdim=False)

        q4_out = []
        q4_out.extend([net(c4_out) for net in self.C4_Q4])
        q4_out.extend([F.interpolate(net(c5_out), scale_factor=2) for net in self.C5_Q4])
        q4_out.extend([F.interpolate(net(q5_out), scale_factor=2) for net in self.Q5_Q4])
        q4_out.extend([F.interpolate(q5_out, scale_factor=2), torch.zeros_like(q4_out[0])])
        q4_out = torch.sum(torch.stack(q4_out, dim=-1) * (self.darts(self.Q4_conn_1) + self.darts(self.Q4_conn_2)),
                           dim=-1, keepdim=False)

        q3_out = []
        q3_out.extend([net(c3_out) for net in self.C3_Q3])
        q3_out.extend([F.interpolate(net(c4_out), scale_factor=2) for net in self.C4_Q3])
        q3_out.extend([F.interpolate(net(q4_out), scale_factor=2) for net in self.Q4_Q3])
        q3_out.extend([F.interpolate(q4_out, scale_factor=2), torch.zeros_like(q3_out[0])])
        q3_out = torch.sum(torch.stack(q3_out, dim=-1) * (self.darts(self.Q3_conn_1) + self.darts(self.Q3_conn_2)),
                           dim=-1, keepdim=False)

        q2_out = []
        q2_out.extend([net(c2_out) for net in self.C2_Q2])
        q2_out.extend([F.interpolate(net(c3_out), scale_factor=2) for net in self.C3_Q2])
        q2_out.extend([F.interpolate(net(q3_out), scale_factor=2) for net in self.Q3_Q2])
        q2_out.extend([F.interpolate(q3_out, scale_factor=2), torch.zeros_like(q2_out[0])])
        q2_out = torch.sum(torch.stack(q2_out, dim=-1) * (self.darts(self.Q2_conn_1) + self.darts(self.Q2_conn_2)),
                           dim=-1, keepdim=False)

        p5_out = []
        p5_out.extend([net(q5_out) for net in self.Q5_P5])
        p5_out.extend([net(c5_out) for net in self.C5_P5])
        p5_out.extend([q5_out, torch.zeros_like(q5_out)])
        p5_out = torch.sum(torch.stack(p5_out, dim=-1) * (self.darts(self.P5_conn_1) + self.darts(self.P5_conn_2)),
                           dim=-1, keepdim=False)

        p4_out = []
        p4_out.extend([net(q4_out) for net in self.Q4_P4])
        p4_out.extend([net(c4_out) for net in self.C4_P4])
        p4_out.extend([F.interpolate(net(q5_out), scale_factor=2) for net in self.Q5_P4])
        p4_out.extend([F.interpolate(net(c5_out), scale_factor=2) for net in self.C5_P4])
        p4_out.extend([q4_out, F.interpolate(q5_out, scale_factor=2), torch.zeros_like(q4_out)])
        p4_out = torch.sum(torch.stack(p4_out, dim=-1) * (self.darts(self.P4_conn_1) + self.darts(self.P4_conn_2)),
                           dim=-1, keepdim=False)

        p3_out = []
        p3_out.extend([net(q3_out) for net in self.Q3_P3])
        p3_out.extend([net(c3_out) for net in self.C3_P3])
        p3_out.extend([F.interpolate(net(q4_out), scale_factor=2) for net in self.Q4_P3])
        p3_out.extend([F.interpolate(net(c4_out), scale_factor=2) for net in self.C4_P3])
        p3_out.extend([q3_out, F.interpolate(q4_out, scale_factor=2), torch.zeros_like(q3_out)])
        p3_out = torch.sum(torch.stack(p3_out, dim=-1) * (self.darts(self.P3_conn_1) + self.darts(self.P3_conn_2)),
                           dim=-1, keepdim=False)

        p2_out = []
        p2_out.extend([net(q2_out) for net in self.Q2_P2])
        p2_out.extend([net(c2_out) for net in self.C2_P2])
        p2_out.extend([F.interpolate(net(q3_out), scale_factor=2) for net in self.Q3_P2])
        p2_out.extend([F.interpolate(net(c3_out), scale_factor=2) for net in self.C3_P2])
        p2_out.extend([q2_out, F.interpolate(q3_out, scale_factor=2), torch.zeros_like(q2_out)])
        p2_out = torch.sum(torch.stack(p2_out, dim=-1) * (self.darts(self.P2_conn_1) + self.darts(self.P2_conn_2)),
                           dim=-1, keepdim=False)

        out_list = [p2_out, p3_out, p4_out, p5_out]
        if self.training:
            self.temperature *= (1 + self.annealing_rate)
        ############ Feature Pyramid ############

        return out_list
