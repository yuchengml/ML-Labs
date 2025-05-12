# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root:
            if root == p:
                return p
            elif root == q:
                return q
            else:
                _p = self.lowestCommonAncestor(root.left, p, q)
                _q = self.lowestCommonAncestor(root.right, p, q)
                if _p and _q:
                    return root
                elif _p and not _q:
                    return _p
                elif _q and not _p:
                    return _q
                else:
                    return None
        else:
            return None
