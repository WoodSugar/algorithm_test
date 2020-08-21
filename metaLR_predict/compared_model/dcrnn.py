# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/17

@Author : Shen Fang
"""
import torch
import random
import torch.nn as nn
import torch.nn.init as init
from utils import fast_power


class DiffConv(nn.Module):
    """
    Class of Diffusion Convolution Operation.
    """
    def __init__(self, in_c, out_c, K):
        """
        :param in_c: int, number of channels of input data.
        :param out_c: int, number of channels of output data.
        :param K:
        """
        super(DiffConv, self).__init__()
        assert K > 0
        self.K = K
        self.in_c = in_c
        self.out_c = out_c
        self.weight_o = nn.Parameter(torch.Tensor(K, in_c, out_c))
        self.weight_i = nn.Parameter(torch.Tensor(K, in_c, out_c))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.weight_o)
        init.xavier_normal_(self.weight_i)

    def forward(self, inputs, graph):
        """
        :param inputs: torch.tensor of size [B, N, C], input features.
        :param graph: torch.tensor of size [N, N], graph structure.
        :return:
            out:, torch.tensor of size [B, N, D], output features.
        """
        deg_i_inv, deg_o_inv = self.norm(graph)
        graph_o = torch.mm(deg_o_inv, graph)  # [N, N]
        graph_i = torch.mm(deg_i_inv, graph)  # [N, N]

        out = 0.

        for k in range(self.K):
            y_o = torch.matmul(torch.matmul(fast_power(graph_o, k), inputs), self.weight_o[k])
            y_i = torch.matmul(torch.matmul(fast_power(graph_i, k), inputs), self.weight_i[k])
            out += (y_o + y_i)

        return out

    @staticmethod
    def norm(graph):
        """
        :param graph: torch.tensor of size [N, N]
        :return:
            deg_i_inv： torch.tensor of size [N, N], a diagonal matrix of the graph input deg.
            deg_o_inv： torch.tensor of size [N, N], a diagonal matrix of the graph output deg.
        """

        deg_o = torch.sum(graph, dim=1, keepdim=False)  # [N]
        deg_i = torch.sum(graph, dim=0, keepdim=False)  # [N]

        deg_o_inv = deg_o.pow(-1)
        deg_o_inv[deg_o_inv == float("inf")] = 0.
        deg_o_inv = torch.diag(deg_o_inv)

        deg_i_inv = deg_i.pow(-1)
        deg_i_inv[deg_i_inv == float("inf")] = 0.
        deg_i_inv = torch.diag(deg_i_inv)
        return deg_i_inv, deg_o_inv


class DCRNNCell(nn.Module):
    """
    Class of the cell operation in the DCRNN.
    """
    def __init__(self, num_nodes, in_c, hid_c, K, bias):
        """
        :param num_nodes: int, Number of nodes in graph.
        :param in_c: int, Number of channels of input tensor.
        :param hid_c: int, Number of channels of hidden state.
        :param K: int, Order of neighbors in graph convolution.
        :param bias: Bool, Whether or not to use bias.
        """
        super(DCRNNCell, self).__init__()
        self.num_nodes = num_nodes
        self.in_channel = in_c
        self.hidden_channel = hid_c
        self.K = K
        self.bias = bias

        self.conv_r = DiffConv(in_c + hid_c, hid_c, K)
        self.conv_z = DiffConv(in_c + hid_c, hid_c, K)
        self.conv_n_1 = DiffConv(in_c, hid_c, K)
        self.conv_n_2 = DiffConv(hid_c, hid_c, K)

    def forward(self, inputs, hid_state, graph):
        """
        :param inputs: torch.tensor of size [B, N, C], input features.
        :param hid_state: torch.tensor of size [B, N, H], hidden states.
        :param graph: torch.tensor of size [N, H], graph structure.
        :return:
            hid_state, torch.tensor of size [B, N, H], updated hidden states.
        """
        combined = torch.cat([inputs, hid_state], dim=-1)  # [B, N, C+H]
        # combined_conv = self.conv_rz(combined, graph)  # [B, N, 2*H]
        # gru_r, gru_z = torch.split(combined_conv, self.hidden_channel, dim=-1)  # 2*[B, N, H]

        gru_r = self.conv_r(combined, graph)
        gru_z = self.conv_z(combined, graph)

        gru_r, gru_z = torch.sigmoid(gru_r), torch.sigmoid(gru_z)

        gru_n = self.conv_n_1(inputs, graph) + gru_r * self.conv_n_2(hid_state, graph)
        gru_n = torch.tanh(gru_n)

        hid_state = (1 - gru_z) * gru_n + gru_z * hid_state

        return hid_state

    def init_hidden(self, B, device):
        """
        :param B: int, batch size.
        :param device: torch.device().
        :return:
            torch.tensor of size [B, N, H], hidden states.
        """
        return torch.zeros(B, self.num_nodes, self.hidden_channel, device=device, dtype=torch.float)


class DCRNN(nn.Module):
    """
    Class of DCRNN model, based on the DCRNN cell module.
    """
    def __init__(self, num_nodes, in_c, hid_c, K, num_layers,
                 bias=True, return_all_layers=False):
        """
        :param num_nodes: int, number of nodes in graph.
        :param in_c: int, number of channels of input data.
        :param hid_c: int, number of channels of hidden states.
        :param K: int, number of hops.
        :param num_layers: int, number of layers.
        :param bias: bool, Whether of use bias.
        :param return_all_layers: bool, whether to return the hidden states of all layers.
        """
        super(DCRNN, self).__init__()
        K = DCRNN._extend_for_multi_layers(K, num_layers)
        hid_c = DCRNN._extend_for_multi_layers(hid_c, num_layers)

        self.cell_list = nn.ModuleList([DCRNNCell(num_nodes, in_c if i == 0 else hid_c[i - 1], hid_c[i], K[i], bias)
                                        for i in range(num_layers)])

        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

    def forward(self, inputs, graph, hid_states=None):
        """
        :param inputs: torch.tensor of size [B, N, T, C], input features.
        :param graph: torch.tensor of size [N, N], graph structure.
        :param hid_states: None, hidden states.
        :return:
            layer_output_list: list, if not return_all_layers, [B, N, T, H] * 1, else [B, N, T, H] * num_layers.
            last_state_list: list, if not return_all_layers, [B, N, H] * 1, else [B, N, H] * num_layers.
        """
        B = inputs.size(0)
        if hid_states is not None:
            hidden_state = hid_states
        else:
            hidden_state = self._init_hidden(B, inputs.device)

        T = inputs.size(2)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = inputs

        for layer_i in range(self.num_layers):
            h = hidden_state[layer_i]
            output_inner = []
            for t in range(T):
                h = self.cell_list[layer_i](cur_layer_input[:, :, t],  h, graph)  # [B, N, H]
                output_inner.append(h.unsqueeze(-2))
            layer_output = torch.cat(output_inner, dim=-2)  # [B, N, T, H]

            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    @staticmethod
    def _extend_for_multi_layers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def _init_hidden(self, B, device):
        init_state = [self.cell_list[i].init_hidden(B, device) for i in range(self.num_layers)]

        return init_state


class DCRNN_Seq2Seq(nn.Module):
    def __init__(self, num_nodes, in_c, hid_c, K, num_layers):
        super(DCRNN_Seq2Seq, self).__init__()
        self.encoder = DCRNN(num_nodes, in_c, hid_c, K, num_layers, return_all_layers=True)
        self.decoder = DCRNN(num_nodes, in_c, hid_c, K, num_layers, return_all_layers=True)
        self.fc_out = nn.Linear(hid_c, in_c)


    def forward(self, input_data, **kwargs):
        device = kwargs["device"]
        graph = input_data["graph"][0].to(device)

        source = input_data["flow_d0_x"].to(device)
        target = input_data["flow_y"].to(device)

        B, N, TRG_len, C = target.size()

        teacher_ration = kwargs["ration"]

        encoder_results = self.encoder(source, graph)
        # tuple: ([B, N, SRC_len, H] * num_layers, [B, N, H] * num_layers.)

        encoder_output = encoder_results[0][-1]  # [B, N, SRC_len, H]
        hidden = encoder_results[1]  # [B, N, H] * num_layers

        decoder_input = source[:, :, -1].unsqueeze(2)  # [B, N, 1, C]

        decoder_output = torch.zeros(B, N, TRG_len, C).to(device)

        for i in range(TRG_len):
            decoder_results = self.decoder(decoder_input, graph, hidden)
            # tuple: ([B, N, 1, H] * num_layers, [B, N, H] * num_layers)

            decoder_out = decoder_results[0][-1]  # [B, N, 1, H]
            hidden = decoder_results[1]  # [B, N, H] * num_layers

            decoder_output[:, :, i] = self.fc_out(decoder_out.squeeze(2))  # [B, N, C]

            decoder_input = target[:, :, i] if random.random() < teacher_ration else decoder_output[:, :, i]

            decoder_input = decoder_input.unsqueeze(2)

        return decoder_output, target
