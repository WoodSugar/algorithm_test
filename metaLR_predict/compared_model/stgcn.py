# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/18

@Author : Shen Fang
"""
import torch
import torch.nn as nn
from model_utils import CausalConv1d
from compared_model.chebnet import ChebConv


class TemporalConv(nn.Module):
    """
    Class of Temporal Convolution.
    """
    def __init__(self, in_c, out_c, residual=True):
        """
        :param in_c: number of channels of input data.
        :param out_c: number of channels of output data.
        :param residual: whether to add the residual connection.
        """
        super(TemporalConv, self).__init__()
        self.temporal_conv = CausalConv1d(in_c, 2 * out_c, k=3, d=1, stride=1, group=1, bias=True)

        self.act = nn.GLU()

        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c:
            self.residual = lambda x: x
        else:
            self.residual = CausalConv1d(in_c, out_c, k=3, d=1)

    def forward(self, inputs):
        """
        :param inputs: torch.tensor of size [B, N, T, C].
        :return:
            output: [B, N, T, D]
        """
        N = inputs.size(1)
        outputs = torch.cat([self.temporal_conv(inputs[:, s_i].permute(0, 2, 1)).unsqueeze(1)
                             for s_i in range(N)], dim=1).permute(0, 1, 3, 2)
        outputs = self.act(outputs)

        residuals = torch.cat([self.residual(inputs[:, s_i].permute(0, 2, 1)).unsqueeze(1)
                               for s_i in range(N)], dim=1).permute(0, 1, 3, 2)

        return outputs + residuals


class SpatialConv(nn.Module):
    """
    Class of spatial convolution operation.
    """
    def __init__(self, in_c, out_c, K, normalize):
        """
        :param in_c: number of channels of input data.
        :param out_c: number of channels of output data.
        :param K: kernel size of convolution.
        :param normalize: whether to use the normalized graph laplacian.
        """
        super(SpatialConv, self).__init__()
        self.s_conv = ChebConv(in_c, out_c, K, normalize=normalize)
        self.act = nn.ReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: torch.tensor of size [B, N, T, C].
        :param graph: torch.tensor of size [N, N].
        :return:
            output: [B, N, T, D]
        """
        T = inputs.size(2)
        output = torch.cat([self.s_conv(inputs[:, :, t_i], graph).unsqueeze(2)
                            for t_i in range(T)], dim=2)
        return self.act(output)


class STBlock(nn.Module):
    def __init__(self, in_c, hid_c, out_c, ks, normalize):
        """
        :param in_c: number of channels of input data.
        :param hid_c: number of channels of hidden features.
        :param out_c: number of channels of output data.
        :param ks: kernel size of spatial convolution.
        :param normalize: whether to use the normalized graph laplacian.
        """
        super(STBlock, self).__init__()
        self.t_conv1 = nn.Conv2d(in_c, hid_c, (1, 3), padding=(0, 1))  # TemporalConv(in_c, hid_c)
        self.s_conv = SpatialConv(hid_c, hid_c, ks, normalize)
        self.t_conv2 = nn.Conv2d(hid_c, out_c, (1, 3), padding=(0, 1))  # TemporalConv(hid_c, out_c)

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, T, C]
        :param graph: [N, N]
        :return:
            output: [B, N, T, D]
        """
        inputs = inputs.permute(0, 3, 1, 2)
        outputs = self.t_conv1(inputs)  # [B, D, N, T]

        outputs = outputs.permute(0, 2, 3, 1)  # [B, N, T, D]
        outputs = self.s_conv(outputs, graph)

        outputs = outputs.permute(0, 3, 1, 2)
        outputs = self.t_conv2(outputs)

        outputs = outputs.permute(0, 2, 3, 1)
        return outputs


class STGCNSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, ks, normalize):
        super(STGCNSubNet, self).__init__()
        self.st_1 = STBlock(in_c, hid_c, hid_c, ks, normalize)
        self.st_2 = STBlock(hid_c, hid_c, out_c, ks, normalize)

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, T, C]
        :param graph: [N, N]
        :return: [B, N, T, D]
        """
        output = self.st_1(inputs, graph)
        output = self.st_2(output, graph)

        return output


class STGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, ks, src_len, trg_len, normalize):
        super(STGCN, self).__init__()
        self.n_stgcn = STGCNSubNet(in_c, hid_c, out_c, ks, normalize)

        self.fc_out = nn.Linear(src_len, trg_len)

    def forward(self, input_data, **kwargs):
        device = kwargs["device"]

        graph = input_data["graph"][0].to(device)  # [N, N]

        source = input_data["flow_d0_x"].to(device)   # [B, N, SRC_len, C]
        target = input_data["flow_y"].to(device)  # [B, N, TRG_len, C]

        features = self.n_stgcn(source, graph)  # [B, N, SRC_len, C]

        predict = self.fc_out(features.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return predict, target
