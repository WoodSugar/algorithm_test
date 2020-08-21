# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/18

@Author : Shen Fang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import MLP


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Simple Graph attention layer.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.F = F.softmax

        self.W = MLP([in_channels, out_channels], act_type=None)
        self.b = nn.Parameter(torch.Tensor(out_channels))
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]

        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)

        return torch.bmm(attention, h) + self.b



class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()

        self.attentions_1 = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.reduce_1 = GraphAttentionLayer(hid_c*n_heads, hid_c)
        # self.norm_1 = SwitchNorm1d(hid_c)
        self.attentions_2 = nn.ModuleList([GraphAttentionLayer(hid_c, hid_c) for _ in range(n_heads)])
        # self.reduce_2 = GraphAttentionLayer(hid_c * n_heads, hid_c)

        # self.attentions_3 = nn.ModuleList([GraphAttentionLayer(hid_c, hid_c) for _ in range(n_heads)])
        # self.reduce_3 = GraphAttentionLayer(hid_c * n_heads, hid_c)

        self.out_att = GraphAttentionLayer(hid_c*n_heads, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = self.act(torch.cat([att(inputs, graph) for att in self.attentions_1], dim=-1))
        outputs = self.reduce_1(outputs, graph)

        outputs = self.act(torch.cat([att(outputs, graph) for att in self.attentions_2], dim=-1))
        # outputs = self.reduce_2(outputs, graph)

        # outputs = self.act(torch.cat([att(outputs, graph) for att in self.attentions_3], dim=-1))

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GAT, self).__init__()
        self.n_subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, input_data, **kwargs):
        device = kwargs["device"]

        graph = input_data["graph"][0].to(device)  # [N, N]

        source = input_data["flow_d0_x"].to(device)   # [B, N, SRC_len, C]
        target = input_data["flow_y"].to(device)  # [B, N, TRG_len, C]

        B, N, SRC_Len, C = source.size()
        TRG_len = target.size(2)

        source = source.view(B, N, -1)  # [B, N, D]

        prediction = self.n_subnet(source, graph).view(B, N, TRG_len, -1)  # [B, N, 1, C]

        return prediction, target
