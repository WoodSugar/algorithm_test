# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/2

@Author : Shen Fang
"""


class PalindromeList:
    def chkPalindrome(self, A):
        # write code here
        # 快慢指针两个
        slow = A.next
        fast = A.next.next

        # 找到中点
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        # 以中点后一节点为起始，反转链表，记得要保存新头结点，因为最后要恢复链表结构
        R = self.reverseNode(slow.next)
        temp_R = R

        # 将两部分链表断开，左右两结点同时开找
        slow.next = None
        L = A

        Flag = True
        while R.next and L.next:
            if R.val != L.val:
                Flag = False
                break
            R = R.next
            L = L.next

        # 最后将反转的右半部分反转回来
        slow.next = self.reverseNode(temp_R)

        return Flag

    def reverseNode(self, head):
        if not head or not head.next:
            return head
        last = self.reverseNode(head.next)
        head.next.next = head
        head.next = None
