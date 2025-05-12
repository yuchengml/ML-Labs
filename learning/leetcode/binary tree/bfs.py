from collections import deque


class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def BFS(root):
    if root is None:
        return

    queue = deque([root])

    while queue:
        node = queue.popleft()  # dequeue a node from the front of the queue
        print(node.value, end=' ')  # visit the node (print its value in this case)

        # enqueue left child
        if node.left:
            queue.append(node.left)
        # enqueue right child
        if node.right:
            queue.append(node.right)
