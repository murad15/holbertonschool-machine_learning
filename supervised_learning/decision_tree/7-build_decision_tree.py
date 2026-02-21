#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


class Node:
    """Internal split node representing a decision point."""

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
            """Check if features > lower bounds."""
            comparisons = [np.greater(x[:, key], self.lower[key])
                           for key in self.lower.keys()]
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            """Check if features <= upper bounds."""
            comparisons = [np.less_equal(x[:, key], self.upper[key])
                           for key in self.upper.keys()]
            return np.all(comparisons, axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Recursive prediction for a single individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

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

    def max_depth_below(self):
        """Get max depth in subtree."""
        lc, rc = self.left_child, self.right_child
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

    def pred(self, x):
        """Return leaf value."""
        return self.value

    def update_bounds_below(self):
        """Leaf nodes stop the recursion."""
        pass

    def get_leaves_below(self):
        """Return self as leaf."""
        return [self]

    def max_depth_below(self):
        """Return leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 for leaf."""
        return 1

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
        self.predict = None

    def np_extrema(self, arr):
        """Returns the minimum and maximum of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feat_subset = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(feat_subset)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively trains a node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        feat_vals = self.explanatory[:, node.feature]
        left_pop = np.logical_and(node.sub_population,
                                  feat_vals > node.threshold)
        right_pop = np.logical_and(node.sub_population,
                                   feat_vals <= node.threshold)

        def is_leaf(pop, depth):
            if np.sum(pop) == 0:
                return True
            if np.sum(pop) < self.min_pop or depth >= self.max_depth:
                return True
            return np.unique(self.target[pop]).size == 1

        if is_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if is_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf node."""
        values, counts = np.unique(self.target[sub_population],
                                   return_counts=True)
        leaf_value = values[np.argmax(counts)] if values.size > 0 else 0
        leaf_child = Leaf(leaf_value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()
        self.cou = self.count_nodes(True)
        self.acc = self.accuracy(self.explanatory, self.target)

        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : {self.cou}\n"
                  f"    - Accuracy on training data : {self.acc}")

    def accuracy(self, test_explanatory, test_target):
        """Calculates model accuracy."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def pred(self, x):
        """Root entry for recursive prediction."""
        return self.root.pred(x)

    def update_predict(self):
        """Compute the vectorized prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum([leaf.value * leaf.indicator(A)
                                        for leaf in leaves], axis=0)

    def update_bounds(self):
        """Compute bounds for all nodes."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all tree leaves."""
        return self.root.get_leaves_below()

    def depth(self):
        """Return tree depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return tree node count."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return tree string."""
        return self.root.__str__()
