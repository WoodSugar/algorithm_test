# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/18

@Author : Shen Fang
"""
import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """
    Class of temporal convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
                              stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        """
        :param x: torch.tensor of size [B, C, T, N], input features.
        :param A: torch.tensor of size [K, N, N], graph structure.
        :return:
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding, ),)
            # nn.BatchNorm2d(out_channels),
            # nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),)
                # nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        :param x: torch.tensor of size [B, C, T, N]
        :param A: torch.tensor of size [K, N, N]
        :return:
        """
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


class STGCNModel(nn.Module):
    def __init__(self, in_c, hid_c, out_c, kernel_size, num_stations, edge_importance_weighting):
        super(STGCNModel, self).__init__()
        temporal_kernel_size = kernel_size[0]
        spatial_kernel_size = kernel_size[1]

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_c, hid_c, kernel_size, stride=1, residual=True),
            st_gcn(hid_c, hid_c, kernel_size, stride=1, residual=True),
            st_gcn(hid_c, hid_c, kernel_size, stride=1, residual=True),
            st_gcn(hid_c, out_c, kernel_size, stride=1, residual=True),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(spatial_kernel_size, num_stations, num_stations))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x, graph):
        """
        :param x: torch.tensor of size [B, N, T, C].
        :param graph: torch.tensor of size [K, N, N].
        :return:
            output: [B, N, T, C]
        """
        x = x.permute(0, 3, 2, 1).contiguous()

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, graph * importance)

        x = x.permute(0, 3, 2, 1)    # [B, N, SRC_len, C]

        return x


class STGCNAction(nn.Module):
    def __init__(self, in_c, hid_c, out_c, kernel_size, num_nodes, src_len, trg_len, edge_importance_weighting):
        super(STGCNAction, self).__init__()
        self.n_stgcn = STGCNModel(in_c, hid_c, out_c, kernel_size, num_nodes, edge_importance_weighting)
        self.fc_out = nn.Linear(src_len, trg_len)

        self.s_k = kernel_size[1]
        self.num_nodes = num_nodes

    def forward(self, input_data, **kwargs):
        device = kwargs["device"]

        graph = input_data["graph"][0].to(device).unsqueeze(0)  # [1, N, N]

        graph_0 = torch.eye(self.num_nodes).to(device).unsqueeze(0)  # [1, N, N]

        graph = torch.cat([graph, graph_0], dim=0)  # [2, N, N]

        source = input_data["flow_d0_x"].to(device)   # [B, N, SRC_len, C]
        target = input_data["flow_y"].to(device)  # [B, N, TRG_len, C]

        features = self.n_stgcn(source, graph)  # [B, N, SRC_len, C]

        predict = self.fc_out(features.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return predict, target