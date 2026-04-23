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
from gaussian_renderer import render, render_contrastive_feature, render_with_depth, render_PGSR, render_contrastive_feature_PGSR

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
from sklearn.neighbors import NearestNeighbors

FEATURE_DIM = 32
FEATURE_GAUSSIAN_ITERATION = 10000

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument('--target', default='scene', type=str)

args = get_combined_args(parser)

MODEL_PATH = args.model_path
SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

scale_gate = torch.nn.Sequential(
    torch.nn.Linear(1, 32, bias=True),
    torch.nn.Sigmoid()
)
scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
scale_gate = scale_gate.cuda()

dataset = model.extract(args)

# If use language-driven segmentation, load clip feature and original masks
dataset.need_features = True

# To obtain mask scales
dataset.need_masks = True

scene_gaussians = GaussianModel(dataset.sh_degree)

feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')
bg_color = [0 for i in range(FEATURE_DIM)]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


point_labels_load = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_label_0.8.npy"), allow_pickle=True)).cuda()
# point_labels_load = torch.load(os.path.join(scene.model_path, "points_label_0.8.npy")).cuda()
print(point_labels_load.dtype, type(point_labels_load), point_labels_load.shape)

points = scene_gaussians.get_xyz
# print(points.shape)
views = scene.getTrainCameras()
render_seg_path = os.path.join(scene.model_path, "renders", "seg_parts_best_0.8_")
render_path = os.path.join(scene.model_path, "renders",  "rendered")
gpt_input_path = os.path.join(scene.model_path,  "gpt_input_0.8_bbox_")
makedirs(render_path, exist_ok=True)
makedirs(render_seg_path, exist_ok=True)
makedirs(gpt_input_path, exist_ok=True)

num_labels = len(torch.unique(point_labels_load)) - 1
flag_labels = torch.zeros((num_labels))
label_view_ratios = {i: [] for i in range(num_labels)}
label_view_indices = {i: [] for i in range(num_labels)}

def knn_filled_unlabeled(xyz: torch.Tensor, labels: torch.Tensor, k: int=16):
    noise_mask = labels == -1
    valid_mask = labels >= 0
    
    if noise_mask.sum() == 0 or valid_mask.sum() == 0:
        return labels
    print("nosie points num: ", noise_mask.sum())
    valid_xyz = xyz[valid_mask]
    noise_xyz = xyz[noise_mask]
    
    knn = NearestNeighbors(n_neighbors=min(k, valid_mask.sum()), algorithm='auto', metric="euclidean")
    knn.fit(valid_xyz.detach().cpu().numpy())
    _, idx_np = knn.kneighbors(noise_xyz.detach().cpu().numpy())
    
    idx_torch = torch.from_numpy(idx_np).to(labels.device)  
    neighbor_labels = labels[valid_mask][idx_torch]
    majority_labels = torch.tensor(
        np.array([np.bincount(row).argmax() for row in neighbor_labels.cpu().numpy()]),
        dtype=labels.dtype,
        device=labels.device
    )

    new_labels = labels.clone()
    new_labels[noise_mask] = majority_labels
    
    return new_labels

point_labels_load = knn_filled_unlabeled(points, point_labels_load)
np.save(os.path.join(scene.model_path, "points_label_0.8_filled.npy"), point_labels_load.cpu().numpy())

for idx, view in enumerate(tqdm(views, desc="Seg 3D part")):
    render_indices = torch.where(flag_labels == 0)
    rendering = render_PGSR(view, scene_gaussians, pipeline.extract(args), background)["render"]
    torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    for i in render_indices[0]:
        points_fg_num = torch.zeros((points.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1) # [N, 4]
        camera_coords = (view.world_view_transform.T @ points_h.T).T # [N, 4]
        depths = camera_coords[:, 2:3] 
        depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
        depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            # visualize depth map in 3d
        H, W = view.image_height, view.image_width
        fx = fov2focal(view.FoVx, W)
        fy = fov2focal(view.FoVy, H)
        cx = W / 2 - 0.5
        cy = H / 2 - 0.5
        K = torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]).float().cuda()
            
        u,v = xyz_to_uv(camera_coords[:, :3], K)
            
        mask_valid = (u >= 0) & (u < depth_map.shape[1]) & (v >= 0) & (v < depth_map.shape[0])  # (N,)
        depth_values =  depth_map[v.int()[mask_valid], u.int()[mask_valid]]
        depths_valid = depths[mask_valid].squeeze()
        visible_mask = (depths_valid <=  depth_values) &  (depths_valid >= 0)  # (N,)
        visible_mask_all = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)  # [N]
        visible_mask_all[mask_valid] = visible_mask
        # print(mask_depth_valid.shape)
        # print((point_labels_load == i.item()).shape)
        mask_depth_label_valid = visible_mask_all & (point_labels_load == i)
        ratio = (mask_depth_label_valid.sum() / (point_labels_load == i).sum()).item()
        label_view_ratios[i.item()].append(ratio)
        label_view_indices[i.item()].append(idx)
        
        
flag_labels = torch.zeros((num_labels))
# try:
#     scene_gaussians.roll_back()
# except:
#     pass
        
for i in tqdm(range(num_labels), desc="Render Seg"):
    if flag_labels[i] == 0 and label_view_ratios[i]:  # Not flagged and has records
        max_idx = torch.tensor(label_view_ratios[i]).argmax().item()
        best_view_idx = label_view_indices[i][max_idx]
        best_view = views[best_view_idx]
        
        scene_gaussians.segment(point_labels_load == i)
        render_result = render_PGSR(best_view, scene_gaussians, pipeline.extract(args), background, return_plane=True, return_depth_normal=False)
        rendering_seg = render_result["render"]
        rendering_seg_depth = render_result["plane_depth"]
        torchvision.utils.save_image(rendering_seg, os.path.join(render_seg_path, '{0:05d}'.format(best_view_idx) + '_' + '{0:03d}'.format(i) + ".png"))
        flag_labels[i] = 1
        
        # Original image
        orig_img_path = os.path.join(render_path, '{0:05d}.png'.format(best_view_idx))
        orig_img = Image.open(orig_img_path).convert("RGB")
        orig_tensor = TF.to_tensor(orig_img)
        depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(best_view_idx) + ".npy")
        orig_depth = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)

        # Create mask (based on segment rendering alpha channel or non-black regions)
        mask_vis = rendering_seg.mean(0) > 0.05
        from scipy import ndimage
        mask_vis_np = mask_vis.cpu().numpy().astype(np.uint8)
        labeled, num_features = ndimage.label(mask_vis_np)

        if num_features > 0:
            # Compute the area of each label
            sizes = ndimage.sum(mask_vis_np, labeled, range(1, num_features + 1))
            max_label = (np.argmax(sizes) + 1)  # labels start from 1
            largest_mask = (labeled == max_label)
        else:
            largest_mask = np.zeros_like(mask_vis_np, dtype=bool)

# Convert back to torch
        mask_vis_filtered = torch.from_numpy(largest_mask).to(mask_vis.device)

        mask_depth = (0.1 + orig_depth.squeeze() )>= rendering_seg_depth.squeeze()
        mask =(mask_vis_filtered & mask_depth).cpu().numpy()  # shape: (H, W)

        # Intermediate overlay image
        overlay = orig_tensor.clone()
        overlay[0][mask] = 0.2  # Blue-ish
        overlay[1][mask] = 0.4
        overlay[2][mask] = 1.0
        overlay = overlay * 0.6 + orig_tensor * 0.4  # Blend original image and blue

        # Bounding box image: show only the mask region
        ys, xs = torch.where(mask_vis_filtered)
        if ys.numel() == 0:
            part_tensor = torch.zeros_like(orig_tensor)
        else:
            y_min, y_max = ys.min().item(), ys.max().item()
            x_min, x_max = xs.min().item(), xs.max().item()

            # Optional padding
            pad = 4
            y_min = max(y_min - pad, 0)
            y_max = min(y_max + pad, H - 1)
            x_min = max(x_min - pad, 0)
            x_max = min(x_max + pad, W - 1)

            # Crop the bounding box region
            crop = rendering_seg[:, y_min:y_max+1, x_min:x_max+1]  # (3, h, w)
            c, h, w = crop.shape

            # Compute target size with aspect-ratio-preserving scaling
            scale = min(H / h, W / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize while preserving aspect ratio
            crop_resized = F.interpolate(crop.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

            # Create white or black background canvas
            part_tensor = torch.ones_like(orig_tensor)  # White background
            # part_tensor = torch.zeros_like(orig_tensor)  # Black background

            # Compute centered position
            top = (H - new_h) // 2
            left = (W - new_w) // 2

            # Paste
            part_tensor[:, top:top+new_h, left:left+new_w] = crop_resized
        # part_tensor = torch.zeros_like(orig_tensor)
        part_tensor = part_tensor.cpu()
        gap_width = 20
        H, W = orig_tensor.shape[1:]

        # Create gap
        gap = torch.ones((3, H, gap_width), dtype=orig_tensor.dtype, device=orig_tensor.device)  # White spacing

        # Insert gaps when concatenating
        grid = torch.cat([gap, orig_tensor, gap, overlay, gap, part_tensor, gap], dim=2)

        # Convert to PIL image
        grid_pil = TF.to_pil_image(grid)

        # Add text
        overlay_x0 = gap_width + W + gap_width   # x start of second column (overlay) in padded image
        overlay_y0 = 0                           # y start is always 0

        abs_x_min = overlay_x0 + x_min
        abs_x_max = overlay_x0 + x_max
        abs_y_min = overlay_y0 + y_min
        abs_y_max = overlay_y0 + y_max

        draw = ImageDraw.Draw(grid_pil)
        draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)],
                       outline=(255, 0, 0), width=4)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=35)

        H = orig_tensor.shape[1]
        W = orig_tensor.shape[2]
        spacing = 5

        # Add white background to extend image height for text (optional)
        # Paste grid_pil onto the white background padded image
        padded = Image.new("RGB", (grid_pil.width, grid_pil.height + 50), (255, 255, 255))
        padded.paste(grid_pil, (0, 0))

# Draw text on the padded image instead
        draw = ImageDraw.Draw(padded)
        labels = ["Original Image", "Mask Overlay", f"Part Image {i}"]
        x_offsets = [gap_width, W + 2*gap_width, 2 * (W + 2*gap_width)]

        for j in range(3):
            text = labels[j]
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            x = x_offsets[j] + (W - text_width) // 2
            y = H + 4
            draw.text((x, y), text, fill="black", font=font)


        # Save
        padded.save(os.path.join(gpt_input_path, '{0:03d}.png'.format(i)))

        # flag_labels[i] = 1
        try:
            scene_gaussians.roll_back()
        except:
            pass
print(torch.where(flag_labels == 0)[0].shape)   

