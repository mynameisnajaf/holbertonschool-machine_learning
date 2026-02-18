#!/usr/bin/env python3
"""A module to build a tree"""
import numpy as np


class Node:
    """A class to represent a node in a decision tree"""

    def __init__(self,
                 feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False, depth=0):
        """Constructor method"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the decision tree"""
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below()
            )

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def __str__(self):
        """string representation of a node in the decision tree"""
        if self.is_leaf:
            return f"{self.left_child}\n"
        else:
            left_str = f'{self.left_child}' if self.left_child else ''
            right_str = f'{self.right_child}' if self.right_child else ''
            return (f"[feature={self.feature}, threshold={self.threshold}]\n" +
                    self.left_child_add_prefix(left_str) +
                    self.right_child_add_prefix(right_str))

    def left_child_add_prefix(self, text):
        """Add a prefix to the left of the decision tree"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add a prefix to the right of the decision tree"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text


class Leaf(Node):
    """A class to represent a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the decision tree"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        return 1

    def __str__(self):
        """Return a string representation of the decision tree"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """A class to represent a decision tree"""

    def __init__(self,
                 max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None
                 ):
        """Constructor method"""
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
        """Return the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the decision tree"""
        return self.root.__str__()
