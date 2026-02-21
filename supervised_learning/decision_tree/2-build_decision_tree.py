#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


class Node:
    """qwdwdq wdq qwd w dqw w"""
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """qwdwdq wdq qwd w dqw w"""
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """qw asd ad asd ad eqweqwe"""
        count = 0

        # Count current node if we are counting all nodes
        if not only_leaves:
            count = 1

        # Recurse left
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)

        # Recurse right
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def left_child_add_prefix(self, text):
        """Add left branch prefix."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |      " + line + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """Add right branch prefix."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "           " + line + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """Return string representation of a node."""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            left_text = str(self.left_child)
            text += "\n" + self.left_child_add_prefix(left_text)

        if self.right_child is not None:
            right_text = str(self.right_child)
            text += "\n" + self.right_child_add_prefix(right_text)

        return text


class Leaf(Node):
    """qwdwdq wdq qwd w dqw w"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """qwdwdq wdq qwd w dqw w"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """qwdwdq wdq qwd w dqw w"""
        return 1

    def __str__(self):
        """qweq eqe qw eqw eqe qw e"""
        return (f"-> leaf [value={self.value}]")

class Decision_Tree():
    """qwdwdq wdq qwd w dqw w"""
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """qwdwdq wdq qwd w dqw w"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """qwdwdq wdq qwd w dqw w"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """qweq eqe qw eqw eqe qw e"""
        return self.root.__str__()
