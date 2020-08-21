# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/17

@Author : Shen Fang
"""
import torch
import torch.nn as nn
import torch.nn.init as init

from model_utils import act_layer, SwitchNorm1d


def remove_self_loop(graph):
    """
    remove the self loop in the graph.

    :param graph: the graph structure, [N, N].
    :return: graph structure without self loop.
    """
    N = graph.size(0)
    for i in range(N):
        graph[i, i] = 0.
    return graph


def get_laplacian(graph, normalize):
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1/2))
        L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph
    return L


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: number of input channels.
    :param out_c: number of output channels.
    :param K: the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = get_laplacian(remove_self_loop(graph), self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, B, N, C]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian


class ChebSubNet(nn.Module):
    """
    The ChebNet model.

    :param channels: The module channels indicating the graph convolution structure, list.
    :param Ks:
    """
    def __init__(self, channels, Ks, act_type, normalize):
        super(ChebSubNet, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(1, len(channels)):
            self.layers.append(ChebConv(channels[i-1], channels[i], Ks[i-1], normalize=normalize))
            self.norms.append(SwitchNorm1d(channels[i], using_bn=False))
            self.acts.append(act_layer(act_type, inplace=True))

    def forward(self, inputs, graph):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs, graph)
            outputs = self.norms[i](outputs)
            outputs = self.acts[i](outputs)

        return outputs


class ChebNet(nn.Module):
    def __init__(self, channels, Ks, act_type, normalize):
        super(ChebNet, self).__init__()
        self.n_subnet = ChebSubNet(channels, Ks, act_type, normalize)

    def forward(self, input_data, **kwargs):
        device = kwargs["device"]

        graph = input_data["graph"][0].to(device)  # [N, N]

        source = input_data["flow_d0_x"].to(device)   # [B, N, T, C]
        target = input_data["flow_y"].to(device)

        B, N, SRC_len, C = source.size()
        TRG_len = target.size(2)

        source = source.view(B, N, -1)  # [B, N, D]

        predict = self.n_subnet(source, graph).view(B, N, TRG_len, -1)  # [B, N, 1, C]

        return predict, target
