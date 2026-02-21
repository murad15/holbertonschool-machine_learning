#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


class Node:
    """Internal split node."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """Init node attributes."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """Prefix left branch text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Prefix right branch text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return node string."""
        label = "root" if self.is_root else "node"
        res = f"{label} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            res += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            res += self.right_child_add_prefix(self.right_child.__str__())
        return res.rstrip()

    def get_leaves_below(self):
        """Get all leaves in subtree."""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def max_depth_below(self):
        """Get max depth in subtree."""
        lc = self.left_child
        rc = self.right_child
        left = lc.max_depth_below() if lc else self.depth
        right = rc.max_depth_below() if rc else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes in subtree."""
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count


class Leaf(Node):
    """Terminal leaf node."""

    def __init__(self, value, depth=None):
        """Init leaf node."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return leaf string."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return self as leaf."""
        return [self]

    def max_depth_below(self):
        """Return leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 for leaf."""
        return 1


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Init tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def __str__(self):
        """Return tree string."""
        return self.root.__str__()

    def get_leaves(self):
        """Return all tree leaves."""
        return self.root.get_leaves_below()

    def depth(self):
        """Return tree depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return tree node count."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
