# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""
while 1:
    try:
        n = int(input())
        data = input().split()
        method = int(input())
        data = sorted(data, key=lambda x : int(x), reverse=(method == 1))
        print(" ".join(data))
    except:
        break