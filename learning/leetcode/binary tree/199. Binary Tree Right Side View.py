# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        outputs = []
        if root is None:
            return outputs

        if root.left is None and root.right is None:
            return [root.val]

        queue = deque([root])
        while queue:
            child_queue = deque()
            while queue:
                node = queue.popleft()  # dequeue a node from the front of the queue
                # print(node.val, end=' ')  # visit the node (print its value in this case)

                # enqueue left child
                if node.left:
                    child_queue.append(node.left)
                # enqueue right child
                if node.right:
                    child_queue.append(node.right)

                prev = node

            outputs.append(prev.val)
            queue = child_queue

        return outputs
