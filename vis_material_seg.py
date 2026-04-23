import os
import argparse
import numpy as np
import matplotlib as mpl
import json
import yaml


def create_material_mapping(material_library):
    return {mat: idx for idx, mat in enumerate(material_library)}

DEFAULT_MATERIAL_LIBRARY = ["wood", "sand", "leather", "plastic", "glass", "fabric", "foam", "food", "composite", 
                            "paper", "metal", "plant", "stone", "soil", "concrete", "cement", "clay", "ceramic", "sky"]

# Initialize global material mapping
MATERIAL_MAPPING = create_material_mapping(DEFAULT_MATERIAL_LIBRARY)
        
DEFAULT_BURNABILITY_LIBRARY = ["burnable", "unburnable"]
BURNABILITY_MAPPING = create_material_mapping(DEFAULT_BURNABILITY_LIBRARY)

MODEL_PATH = None  # Set from command line args
CONFIG_FILE = None  # Set from command line args

import os
import json
import re
from sklearn.neighbors import NearestNeighbors

def update_material_and_burnability(points, materials, burnability, k=10):
    """
    Update each point's material and burnability based on KNN.

    Args:
    points -- Input points, tensor of shape (N, 3).
    point_labels_load -- Labels for each point, including material and burnability, tensor of shape (N, 2).
    k -- Number of nearest neighbors for KNN lookup.

    Returns:
    material_ids, burnability_ids -- Updated material and burnability.
    """
    # Extract points and label data
    # point_labels = point_labels_load.numpy()  # Convert to NumPy array for processing
    materials = materials.cpu().numpy()  # Assuming material is the first column
    burnability = burnability.cpu().numpy()  # Assuming burnability is the second column

    # Use KNeighborsClassifier to find nearest neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(points)  # Fit point data

    # Get k nearest neighbors for each point
    _, indices = knn.kneighbors(points)

    # Update material and burnability
    updated_materials = np.copy(materials)
    updated_burnability = np.copy(burnability)

    # Update material and burnability for each point
    for i in range(len(points)):
        # For each point i, get the materials of k nearest neighbors
        neighbors_materials = materials[indices[i]]
        # Select the most common material among nearest neighbors
        most_common_material = np.bincount(neighbors_materials).argmax()
        updated_materials[i] = most_common_material

        # Now update burnability by selecting from nearest neighbors' burnability
        # Select burnability based on neighbors' material
        neighbors_burnability = burnability[indices[i]]
        # Only select burnability from neighbors with the same material
        neighbors_material_match = neighbors_materials == most_common_material
        filtered_burnability = neighbors_burnability[neighbors_material_match]
        
        if len(filtered_burnability) > 0:
            most_common_burnability = np.bincount(filtered_burnability).argmax()
            updated_burnability[i] = most_common_burnability

    # Convert updated results back to PyTorch tensors
    material_ids = torch.tensor(updated_materials)
    burnability_ids = torch.tensor(updated_burnability)

    return material_ids, burnability_ids

def clean_label_string(s):
    s = s.strip().lower()  # Strip leading/trailing whitespace and convert to lowercase
    s = re.sub(r'[^\w\s]', '', s)  # Remove all punctuation (keep letters, digits, and spaces)
    return s

def parse_label_file_to_json(txt_path, material_mapping, burnability_mapping, output_json_path):
    label_dict = {}

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 4:
                print(f"Skipping malformed line: {line}")
                continue

            img_path, description, material, burnability = parts

            # Extract image number (assuming format like .../120.png)
            filename = os.path.basename(img_path)
            img_number = os.path.splitext(filename)[0]  # Remove .png extension

            material_id = material_mapping.get(clean_label_string(material), -1)
            burnability_id = burnability_mapping.get(clean_label_string(burnability), -1)

            label_dict[img_number] = {
                "material": clean_label_string(material),
                "material_id": material_id,
                "burnanility": clean_label_string(burnability),
                "burnability_id": burnability_id
            }

    # Save as JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=2)

    print(f"Saved label mapping to {output_json_path}")

def parse_label_files_to_json(txt_paths, material_mapping, burnability_mapping, output_json_path):
    label_dict = {}

    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 4:
                    print(f"Skipping malformed line: {line}")
                    continue

                img_path, description, material, burnability = parts

                filename = os.path.basename(img_path)
                img_number = os.path.splitext(filename)[0]

                material_id = material_mapping.get(clean_label_string(material), -1)
                burnability_id = burnability_mapping.get(clean_label_string(burnability), -1)

                label_dict[img_number] = {
                    "material": clean_label_string(material),
                    "material_id": material_id,
                    "burnanility": clean_label_string(burnability),
                    "burnability_id": burnability_id
                }

    # Save as JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=2)

    print(f"Saved merged label mapping to {output_json_path}")


import time
import os
import torch
import pytorch3d.ops
from plyfile import PlyData, PlyElement
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature, render_with_depth

# import utils.contrastive_decoder_utils
from utils.sh_utils import SH2RGB

from tqdm import tqdm
from utils.graphics_utils import fov2focal
from utils.my_utils import depth_to_xyz_map, xyz_to_uv
from os import makedirs
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw, ImageFont
import open3d as o3d

def assign_material_and_burnability_to_points(points, point_labels_load, label_json_path):
    with open(label_json_path, 'r') as f:
        label_info = json.load(f)

    num_points = points.shape[0]
    material_ids = torch.full((num_points,), -1, dtype=torch.int32, device=points.device)
    burnability_ids = torch.full((num_points,), BURNABILITY_MAPPING["unburnable"], dtype=torch.int32, device=points.device)

    for label in torch.unique(point_labels_load).tolist():
        if label == -1:
            print("[Warning: Label = -1]")
            continue
        label_str = str(label).zfill(3)
        if label_str in label_info:
            mask = (point_labels_load == label)
            material_id = label_info[label_str]["material_id"]
            print(f"{label_str}: {material_id}")
            material_ids[mask] = label_info[label_str]["material_id"]
            burnability_ids[mask] = label_info[label_str]["burnability_id"]
    assert (material_ids != -1).all()
    return material_ids, burnability_ids

def save_colored_point_cloud(points, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved point cloud to {save_path}")
    
def get_color_map(num_classes, colormap='tab20'):
    # cmap = plt.cm.get_cmap(colormap, num_classes)  # Get gradient color bar from 'gist_rainbow'
    # colors = torch.tensor([cmap(i / (num_classes - 1))[:3] for i in range(num_classes)], dtype=torch.float32)
    #'Set1'
    cmap = plt.cm.get_cmap(colormap, num_classes)  # Choose any colormap, e.g., 'tab20b', 'Set1', 'Paired'
    colors = torch.tensor([cmap(i)[:3] for i in range(num_classes)], dtype=torch.float32)
    return colors

def colorize_points_by_attribute(attribute_ids, num_classes):
    color_map = get_color_map(num_classes)
    return color_map[attribute_ids.cpu()]

def generate_random_colors(num_classes, seed=42):
    np.random.seed(seed)
    colors = np.random.rand(num_classes, 3)  # One RGB triplet per class
    return torch.tensor(colors, dtype=torch.float32)

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)


FEATURE_DIM = 32
FEATURE_GAUSSIAN_ITERATION = 10000

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--target', default='scene', type=str)
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file")
    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)

    MODEL_PATH = args.model_path
    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
    scale_gate = scale_gate.cuda()


    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = True

    scene_gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')
    bg_color = [0 for i in range(FEATURE_DIM)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_json_path = os.path.join(scene.model_path, "label_mapping_hierachical.json")

    parse_label_files_to_json(
        txt_paths=[os.path.join(scene.model_path, "gpt_input/gpt_input_mul/gpt_input_mul_gpttool.txt"), os.path.join(scene.model_path, "gpt_input/gpt_input_single/gpt_input_single_gpttool.txt")],
        material_mapping=MATERIAL_MAPPING,
        burnability_mapping=BURNABILITY_MAPPING,
        output_json_path=output_json_path
    )

    points = scene_gaussians.get_xyz
    point_labels_load = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_label_mulscale.npy"))).cuda()
    material_ids, burnability_ids = assign_material_and_burnability_to_points(
        points, point_labels_load, output_json_path
    )
    np.save(os.path.join(scene.model_path, "points_materials.npy"), material_ids.cpu().numpy())
    np.save(os.path.join(scene.model_path, "points_burnability.npy"), burnability_ids.cpu().numpy())

    material_colors = colorize_points_by_attribute(material_ids, len(MATERIAL_MAPPING))
    save_colored_point_cloud(points, material_colors, os.path.join(scene.model_path, "material_colored_hierach.ply"))

    burn_colors = colorize_points_by_attribute(burnability_ids, len(BURNABILITY_MAPPING))
    save_colored_point_cloud(points, burn_colors, os.path.join(scene.model_path, "burnability_colored_hierach.ply"))

    material_id_updated, burn_id_updated = update_material_and_burnability(points.detach().cpu().numpy(), material_ids, burnability_ids)
    np.save(os.path.join(scene.model_path, "points_materials_knn.npy"), material_id_updated.cpu().numpy())
    np.save(os.path.join(scene.model_path, "points_burnability_knn.npy"), burn_id_updated.cpu().numpy())
    material_colors_updated = colorize_points_by_attribute(material_id_updated, len(MATERIAL_MAPPING))
    burn_colors_updated = colorize_points_by_attribute(burn_id_updated, len(BURNABILITY_MAPPING))
    save_colored_point_cloud(points, material_colors_updated, os.path.join(scene.model_path, "material_colored_hierach_knn.ply"))
    save_colored_point_cloud(points, burn_colors_updated, os.path.join(scene.model_path, "burnability_colored_hierach_knn.ply"))