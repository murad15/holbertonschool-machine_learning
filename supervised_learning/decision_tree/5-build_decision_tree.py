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
        self.lower = None
        self.upper = None
        self.indicator = None

    def update_indicator(self):
        """Compute the indicator function based on feature bounds."""
        def is_large_enough(x):
            """Check if all features are > lower bounds."""
            # Creates a boolean array for each feature constraint
            comparisons = [np.greater(x[:, key], self.lower[key])
                           for key in self.lower.keys()]
            # Returns True only if ALL feature constraints are met per row
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            """Check if all features are <= upper bounds."""
            # Creates a boolean array for each feature constraint
            comparisons = [np.less_equal(x[:, key], self.upper[key])
                           for key in self.upper.keys()]
            # Returns True only if ALL feature constraints are met per row
            return np.all(comparisons, axis=0)

        # The indicator is True if the sample is both large and small enough
        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def update_bounds_below(self):
        """Recursively compute feature bounds for all sub-nodes."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

        if self.left_child is not None:
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child is not None:
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def get_leaves_below(self):
        """Recursively gathers all Leaf instances."""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

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


class Leaf(Node):
    """Terminal leaf node containing a prediction value."""

    def __init__(self, value, depth=None):
        """Init leaf node."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaf nodes stop the recursion."""
        pass

    def get_leaves_below(self):
        """Return self as leaf."""
        return [self]

    def __str__(self):
        """Return leaf string."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree container and manager."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Init tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def update_bounds(self):
        """Compute bounds for all nodes."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all tree leaves."""
        return self.root.get_leaves_below()

    def __str__(self):
        """Return tree string."""
        return self.root.__str__()

