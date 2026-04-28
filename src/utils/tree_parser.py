#PURPOSE: Node class and tree parsing utilities for cascade trees
#INPUT: tree files from twitter15/16
#OUTPUT: list of Node objects 

import ast
import math
from collections import Counter, deque
from pathlib import Path


# one node in a cascade tree
class Node:
    def __init__(self, uid, time):
        # set at construction
        self.uid      = uid
        self.time     = time
        self.parent   = None
        self.children = []
        # set during annotate_tree()
        self.depth             = 0
        self.time_since_parent = 0.0
        self.num_siblings      = 0
        self.sibling_rank      = 0
        self.user_post_count   = 1
        # root-only, set in build_data()
        self.text_vec    = None
        self.emotion_vec = None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def num_children(self):
        return len(self.children)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    # 10-dim structural feature vector for this node, normalized by tree-level maxes
    def feature_vector(self, max_depth, max_time, max_fanout):
        safe_fanout = max(max_fanout, 1)
        # some tree files have negative times (replies recorded as pre-dating the source);
        # clamp to 0 so log1p stays defined
        safe_time   = max(self.time, 0.0)
        safe_tsp    = max(self.time_since_parent, 0.0)
        return [
            self.depth / max(max_depth, 1),                # depth of node
            math.log1p(safe_time),                          # absolute time
            safe_time / max(max_time, 1.0),                 # time relative to max time
            self.num_children / safe_fanout,                # children vs. max fanout
            1.0 if self.is_root else 0.0,                   # is root
            1.0 if self.is_leaf else 0.0,                   # is leaf
            math.log1p(safe_tsp),                           # response latency
            self.num_siblings / safe_fanout,                # siblings vs. max fanout
            self.sibling_rank / max(self.num_siblings, 1),  # rank among siblings
            math.log1p(self.user_post_count),               # times this user posted in cascade
        ]


# read a tree file and return the root node with all children linked
#   tree_id  : string tweet id, e.g. "552783667740393472"
#   path_dir : Path to the dataset folder (PATH_15 or PATH_16)
def parse_tree(tree_id, path_dir):
    tree_file = Path(path_dir) / "tree" / f"{tree_id}.txt"
    nodes = {}        # raw tuple -> Node, so we create each node at most once
    root_tuple = None # the source tweet, identified by the ROOT sentinel edge

    with open(tree_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            left, right = line.strip().split("->")
            parent_tuple = tuple(ast.literal_eval(left))
            child_tuple  = tuple(ast.literal_eval(right))

            # ROOT sentinel edge tells us which node is the source tweet
            if parent_tuple[0] == "ROOT":
                root_tuple = child_tuple
                if child_tuple not in nodes:
                    nodes[child_tuple] = Node(child_tuple[0], float(child_tuple[2]))
                continue
            # skip self loops (some trees have node_1 -> node_1)
            if parent_tuple == child_tuple:
                continue
            # never let the root be assigned a parent (handles cyclic edges)
            if child_tuple == root_tuple:
                continue

            if parent_tuple not in nodes:
                nodes[parent_tuple] = Node(parent_tuple[0], float(parent_tuple[2]))
            if child_tuple not in nodes:
                nodes[child_tuple]  = Node(child_tuple[0],  float(child_tuple[2]))

            parent = nodes[parent_tuple]
            child  = nodes[child_tuple]
            # first edge that names this child as a child wins; ignore later cyclic ones
            if child.parent is None:
                child.parent = parent
                parent.children.append(child)

    return nodes[root_tuple]


# BFS traversal that fills in depth, time_since_parent, num_siblings, sibling_rank, and user_post_count
#   root : Node returned by parse_tree()
#   returns list of all nodes in BFS order (root is always index 0)
def annotate_tree(root):
    all_nodes = []
    visited = {root}
    queue = deque([root])
    #BFS traversal to hit each node
    while queue:
        node = queue.popleft()
        all_nodes.append(node)
        # sort children chronologically so sibling_rank is in time order
        node.children.sort(key=lambda n: n.time)
        for rank, child in enumerate(node.children):
            if child in visited:
                continue
            visited.add(child)
            child.depth             = node.depth + 1
            child.time_since_parent = child.time - node.time
            child.num_siblings      = node.num_children - 1
            child.sibling_rank      = rank
            queue.append(child)

    #counter for easy 
    uid_counts = Counter(n.uid for n in all_nodes)
    for node in all_nodes:
        node.user_post_count = uid_counts[node.uid]

    return all_nodes
