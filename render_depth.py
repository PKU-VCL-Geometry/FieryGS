import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_with_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import imageio
from utils.graphics_utils import fov2focal
from utils.my_utils import depth_to_xyz_map, xyz_to_uv
import open3d as o3d
import matplotlib as mpl

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_added_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_added")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(render_added_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background)["render"]
        # print(rendering.shape)
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # gaussians.add_gaussians()
        # rendering_added = render(view, gaussians, pipeline, background)["render"]
        # torchvision.utils.save_image(rendering_added, os.path.join(render_added_path, '{0:05d}'.format(idx) + ".png"))
        
        # # print(render(view, gaussians, pipeline, background)["depth"])
        depth = render_with_depth(view, gaussians, pipeline, background)["depth"][0]
        depth_np = depth.cpu().numpy()
        # print(depth_np.shape)
        # print(depth_np[:2, :2])
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth_np)
        # depth_render = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # depth = depth.permute(1, 2, 0) # (3, h, w) -> (h, w, 3)
        # normal = depth_pcd2normal(depth).permute(2, 0, 1)
        # print(normal.shape)
        # normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                
        # #     # transform to world space
        # c2w = (view.world_view_transform.T).inverse()
        # normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
        # normal = normal2.reshape(3, *normal.shape[1:])
        # normal = (normal + 1.) / 2.
        
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        # for timestamp in range(1, 501):
        #     import copy
        #     gaussians_fire = copy.deepcopy(gaussians)
        #     gaussians_fire.add_gaussians(timestamp, 15)
        #     render_fire(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras()[0:1], gaussians_fire, pipeline, background, timestamp)

        # if not skip_train:
        #     from utils.camera_utils import generate_cam_path
        #     cam_traj = generate_cam_path(scene.getTrainCameras(), n_frames=400)
        #     # render_set(dataset.model_path, "traj_1", scene.loaded_iter, cam_traj, gaussians, pipeline, background)
        #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras()[0:1], gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)