# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/17

@Author : Shen Fang
"""
from utils import plot_curve


x = [0, 500]
y = [0, 60]

rf = "Subway_lr_seq2seq/lr_seq2seq"
plot_curve(result_file=rf, x_range=x, y_range=y)

rf = "Subway_gru_seq2seq/gru_seq2seq"
plot_curve(result_file=rf, x_range=x, y_range=y)

rf = "Subway_chebnet/chebnet"
plot_curve(result_file=rf, x_range=x, y_range=y)
