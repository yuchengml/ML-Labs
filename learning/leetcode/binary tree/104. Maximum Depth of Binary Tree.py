# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def getDepth(self, root: Optional[TreeNode], dep: int) -> int:
        if root:
            return max(self.getDepth(root.left, dep + 1), self.getDepth(root.right, dep + 1))
        else:
            return dep

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        return self.getDepth(root, 0)
