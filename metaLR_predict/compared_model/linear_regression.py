# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/17

@Author : Shen Fang
"""
import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, in_c, src_len, trg_len):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_c * src_len, in_c * trg_len)

    def forward(self, input_data, **kwargs):
        device = kwargs["device"]

        source = input_data["flow_d0_x"].to(device)  # [B, N, SRC_len, C]
        target = input_data["flow_y"].to(device)  # [B, N, TRG_len, C]

        B, N, SRC_Len, C = source.size()
        TRG_Len = target.size(2)

        source = source.view(B, N, -1)

        prediction = self.linear(source)  # [B, N, TRG_Len, C]
        prediction = prediction.view(B, N, TRG_Len, -1)

        return prediction, target
