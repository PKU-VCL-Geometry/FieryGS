import os
import numpy as np
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, FeatureGaussianModel
import json


FEATURE_DIM = 32
FEATURE_GAUSSIAN_ITERATION = 10000

def get_leaf_index_files(tree_json: str, parts_root: str):
    """Return a list of absolute paths to .npy files corresponding to all leaf nodes"""
    with open(tree_json, "r") as f:
        tree = json.load(f)

    leaf_files = []

    def dfs(node):
        if node.get("status") == "leaf" or not node.get("children"):
            leaf_files.append(os.path.join(parts_root, node["point_indices_file"]))
        else:
            for c in node["children"]:
                dfs(c)

    dfs(tree)
    return leaf_files

def check_leaf_partition(leaf_files, total_points: int):
    visited = np.zeros(total_points, dtype=np.bool_)
    overlap = False
    for f in leaf_files:
        idx = np.load(f)
        if visited[idx].any():
            print("[FAIL] overlap in", f)
            overlap = True
        visited[idx] = True

    if visited.all() and not overlap:
        print("[PASS] Full coverage with no overlap")
    else:
        print("[FAIL] Verification failed")
        
def is_leaf(node):
    return node.get("status") == "leaf"

def build_node_lookup(tree):
    lookup = {}
    def dfs(node):
        lookup[node["node_id"]] = node
        for child in node.get("children", []):
            dfs(child)
    dfs(tree)
    return lookup


# parser = ArgumentParser(description="Testing script parameters")
# model = ModelParams(parser, sentinel=True)
# pipeline = PipelineParams(parser)
# parser.add_argument('--target', default='scene', type=str)

# args = get_combined_args(parser, model_path=MODEL_PATH)

# dataset = model.extract(args)

# # If use language-driven segmentation, load clip feature and original masks
# dataset.need_features = False

# # To obtain mask scales
# dataset.need_masks = True

# scene_gaussians = GaussianModel(dataset.sh_degree)

# feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
# scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')

# points = scene_gaussians.get_xyz
# total_points = points.shape[0]
# print("total points number: ", total_points)

# leaf_files = get_leaf_index_files("output/garden/parts_tree/parts_tree.json",
#                                   "output/garden/parts_tree")

# print("leaf count :", len(leaf_files))

# check_leaf_partition(leaf_files, total_points=total_points)

