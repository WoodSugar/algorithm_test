# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def insertionSortList(self , head ):
        # write code here
        if not head or not head.next:
            return head
        cur = ListNode(0)
        cur.next = head
        while head and head.next:
            if head.val < head.next.val:
                head = head.next
                continue
            pre = cur
            while pre.next.val <head.next.val:
                pre = pre.next
            nextnode = head.next
            head.next = nextnode.next
            nextnode.next= pre.next
            pre.next = nextnode
        return cur.next
