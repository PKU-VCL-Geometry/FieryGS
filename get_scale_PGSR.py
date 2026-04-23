import torch


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)
import yaml

import os
FEATURE_DIM = 32

DATA_ROOT = './data/'

ALLOW_PRINCIPLE_POINT_SHIFT = False


def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", required=True, type=str)

    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")
    
    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=-1, shuffle=False, mode='eval', target='scene')


    # Detect mask directory: prefer sam_masks, fall back to sam2_masks
    mask_dir = os.path.join(dataset.source_path, 'sam_masks')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(dataset.source_path, 'sam2_masks')
    assert os.path.exists(mask_dir), f"No mask directory found. Please run extract_sam2_mask.py first. Looked for: {mask_dir}"

    from tqdm import tqdm
    images_masks = {}
    for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(dataset.source_path, 'images'))))):
        # print(image_path)
        image = cv2.imread(os.path.join(os.path.join(dataset.source_path, 'images'), image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = torch.load(os.path.join(mask_dir, image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))
        # N_mask, C

        images_masks[image_path.split('.')[0]] = masks.cpu().float()


    OUTPUT_DIR = os.path.join(args.image_root, 'mask_scales_PGSR')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()

    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device = 'cuda')

    for it, view in tqdm(enumerate(cameras)):

        rendered_pkg = gaussian_renderer.render_PGSR(view, scene_gaussians, pipeline.extract(args), background, app_model=None,
                            return_plane=True, return_depth_normal=False)

        depth = rendered_pkg['plane_depth']

        # plt.imshow(depth.detach().cpu().squeeze().numpy())
        corresponding_masks = images_masks[view.image_name]

        # generate_grid_index(depth.squeeze())[50, 1]
        # Project depth to 3D points in camera space
        depth = depth.cpu().squeeze()

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:,:,-1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(cameras[0].FoVx / 2)
        fy = cy / np.tan(cameras[0].FoVy / 2)


        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

        upsampled_mask = torch.nn.functional.interpolate(corresponding_masks.unsqueeze(1), mode = 'bilinear', size = (depth.shape[0], depth.shape[1]), align_corners = False)

        eroded_masks = torch.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)

        scale = torch.zeros(len(corresponding_masks))
        ## Consider [sigma_x, sigma_y, sigma_z] * 2 of the 3D points corresponding to this mask as the diameter
        for mask_id in range(len(corresponding_masks)):
            
            if (eroded_masks[mask_id] == 1).sum().item() == 0:
                scale[mask_id] == 0.0
                continue
            
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
            
            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

        torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + '.pt'))