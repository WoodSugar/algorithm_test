# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""
import sys
import math

try:
    while True:
        line = sys.stdin.readline()
        number = int(line)
        print(2 ** int(math.log(number, 2)) - 1)

except:
    pass