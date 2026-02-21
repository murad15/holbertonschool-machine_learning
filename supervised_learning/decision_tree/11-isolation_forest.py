#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Something that function does"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Something that function does"""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Something that function does"""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        max = self.max_depth
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=max, seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """ returns the n_suspecve the smallest mean depth """
        # Step 1: Get the mean isolation depth for all individuals
        # Outliers have low depth values.
        depths = self.predict(explanatory)
        # Step 2: Use np.argsort to find indices of the smallest values
        # Outliers are at the beginning of this sorted array.
        indices = np.argsort(depths)
        # Step 3: Select the first n_suspects indices
        suspect_indices = indices[:n_suspects]
        # Step 4: Return the actual rows from the explanatory matrix
        return explanatory[suspect_indices], depths[suspect_indices]


class Isolation_Random_Tree():
    """Tree designed to isolate anomalies based on path depth."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Init attributes."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Same as Decision_Tree."""
        return self.root.__str__()

    def depth(self):
        """Same as Decision_Tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Same as Decision_Tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Same as Decision_Tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Same as Decision_Tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Same as Decision_Tree logic."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        # Predict returns the leaf.value (which is the depth)
        self.predict = lambda A: np.sum([leaf.value * leaf.indicator(A)
                                        for leaf in leaves], axis=0)

    def np_extrema(self, arr):
        """Min and max helper."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Same logic as Decision_Tree."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feat_subset = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(feat_subset)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates leaf where value is the node depth."""
        # For Isolation Trees, the prediction is the depth of the leaf
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Same as Decision_Tree."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursive training based on isolation logic."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feat_vals = self.explanatory[:, node.feature]
        left_population = np.logical_and(node.sub_population,
                                         feat_vals > node.threshold)
        right_population = np.logical_and(node.sub_population,
                                          feat_vals <= node.threshold)

        # Different from Decision_Tree: we only stop at max_depth
        # or when population is too small (usually 1)
        def is_leaf(pop, depth):
            pop_count = np.sum(pop)
            return pop_count <= self.min_pop or depth >= self.max_depth

        # Process Left
        if is_leaf(left_population, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Process Right
        if is_leaf(right_population, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train the Isolation Tree."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        # Corrected initialization of sub_population mask
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")


class Random_Forest():
    """Random Forest classifier using an ensemble of decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Init forest attributes."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class for each sample using majority voting.
        """
        # Collect predictions from all trees: shape (n_trees, n_individuals)
        tree_predictions = np.array([p(explanatory) for p in self.numpy_preds])

        # For each column (individual), find the most frequent value
        # We use a loop-free approach with np.apply_along_axis or mode logic
        def get_mode(column):
            values, counts = np.unique(column, return_counts=True)
            return values[np.argmax(counts)]

        # Applying mode across the tree-axis (axis 0)
        return np.apply_along_axis(get_mode, 0, tree_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Trains the forest by fitting multiple decision trees."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            # Each tree gets a unique seed for variation
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}\n"
                  f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}\n"
                  f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}\n"
                  f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}\n"
                  f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """Calculates the accuracy of the forest predictions."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size


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

    def possible_thresholds(self, node, feature):
        """Returns all middle points between sorted unique feature values."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Finds the best threshold for one feature using Gini impurity."""
        # Data for individuals currently in this node
        sub_expl = self.explanatory[:, feature][node.sub_population]
        sub_target = self.target[node.sub_population]
        thresholds = self.possible_thresholds(node, feature)

        # Unique classes present in the current node
        classes = np.unique(sub_target)

        # Reshape for broadcasting:
        # sub_expl: (n, 1, 1) | thresholds: (1, t, 1) | sub_target: (n, 1, 1)
        # classes: (1, 1, c)
        n_ind = sub_expl.shape[0]
        n_thresh = thresholds.shape[0]
        n_class = classes.shape[0]

        # 1. Indicator for individuals > threshold
        # shape: (n, t) -> (n, t, 1)
        is_left = (sub_expl[:, np.newaxis] > thresholds[np.newaxis, :])

        # 2. Indicator for class membership
        # shape: (n, c) -> (n, 1, c)
        is_class = (sub_target[:, np.newaxis] == classes[np.newaxis, :])
        # 3. Combine: Left_F[i, j, k] is True if individual i is class k
        # and satisfies threshold j. shape: (n, t, c)
        left_f = np.logical_and(is_left[:, :, np.newaxis],
                                is_class[:, np.newaxis, :])

        # Counts for left child
        left_class_counts = np.sum(left_f, axis=0)  # (t, c)
        left_total_counts = np.sum(is_left, axis=0)  # (t,)

        # Counts for right child (total in node - left)
        # We need total counts of each class in the whole node:
        node_class_counts = np.sum(is_class, axis=0).flatten()  # (c,)
        right_class_counts = node_class_counts - left_class_counts  # (t, c)
        right_total_counts = n_ind - left_total_counts  # (t,)

        # To avoid division by zero if a child is empty
        # Gini = 1 - sum((count/total)^2)
        # We rewrite as: Gini = (total^2 - sum(count^2)) / total^2
        # But we actually want weighted Gini: (total/n_ind) * Gini
        # Simplified: (total^2 - sum(count^2)) / (total * n_ind)

        def compute_weighted_gini(class_counts, total_counts):
            """Weighted Gini helper."""
            with np.errstate(divide='ignore', invalid='ignore'):
                sum_sq = np.sum(class_counts**2, axis=1)
                gini = (total_counts - sum_sq / total_counts) / n_ind
                return np.nan_to_num(gini)

        weighted_gini_left = compute_weighted_gini(left_class_counts,
                                                   left_total_counts)
        weighted_gini_right = compute_weighted_gini(right_class_counts,
                                                    right_total_counts)
        gini_total = weighted_gini_left + weighted_gini_right
        best_idx = np.argmin(gini_total)
        return thresholds[best_idx], gini_total[best_idx]

    def Gini_split_criterion(self, node):
        """Finds the best feature and threshold for the node."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        # X[:, 0] are thresholds, X[:, 1] are Gini averages
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
