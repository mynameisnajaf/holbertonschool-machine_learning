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

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    └-- " + lines[0]
        for x in lines[1:]:
            new_text += "\n        " + x
        return (new_text)

    def __str__(self):
        """Return a string representation of the decision tree"""
        text = f"[feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            left_text = str(self.left_child)
            text += "\n" + self.left_child_add_prefix(left_text)

        if self.right_child:
            right_text = str(self.right_child)
            text += "\n" + self.right_child_add_prefix(right_text)

        return text


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

def example_0():
    leaf0 = Leaf(0, depth=1)
    leaf1 = Leaf(0, depth=2)
    leaf2 = Leaf(1, depth=2)
    internal_node = Node(feature=1, threshold=30000, left_child=leaf1, right_child=leaf2, depth=1)
    root = Node(feature=0, threshold=.5, left_child=leaf0, right_child=internal_node, depth=0, is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

#Print Tree example 0
print(example_0())
#Print Tree example 1
print(example_1(4))