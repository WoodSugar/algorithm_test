# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""

class Solution:
    def twoSum(self, numbers, target):
        # write code here
        n = len(numbers)
        hashmap = {}

        for i in range(n):
            if numbers[i] in hashmap:
                return [hashmap[numbers[i]] + 1, i + 1]
            else:
                hashmap[target - numbers[i]] = i
