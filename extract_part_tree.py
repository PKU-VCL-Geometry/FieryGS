# Borrowed from SAGA & GARField, but modified
import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from argparse import ArgumentParser, Namespace
import random
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render_PGSR
import hdbscan
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from os import makedirs
from tqdm import tqdm
import json
import glob
from utils.graphics_utils import fov2focal
from utils.my_utils import xyz_to_uv
from os import makedirs
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw, ImageFont
from typing import Optional 
from utils.tree_utils import *
import yaml

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)   
FEATURE_DIM = 32
FEATURE_GAUSSIAN_ITERATION = 10000

def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()
    # print(scales.max(), '?')

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    
    def quantile_transformer_func(scales):
        scales_shape = scales.shape

        scales = scales.reshape(-1,1)
        
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device).reshape(scales_shape)

    return quantile_transformer_func, quantile_transformer

def get_scaled_features(point_features: torch.Tensor, scale_gate: nn.Module, scale: float):
    with torch.no_grad():
        scale = torch.tensor([scale]).cuda()
        gates = scale_gate(scale)
        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * gates.unsqueeze(0)

        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
    
    return normed_point_features

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

def segment_gaussians_across_scales(
    point_features: torch.Tensor,
    point_xyz: torch.Tensor,
    scale_gate: nn.Module,
    scales: list[float],
    model_path: str = None,
    min_cluster_size: int=10,
    epsilon: float=0.01,
    similarity_threshold: float=0.5,
):
    parts_tree = {}
    parts_tree_path = os.path.join(model_path or MODEL_PATH, "parts_tree")
    makedirs(parts_tree_path, exist_ok=True)
    
    for scale in scales:
        normed_point_features = get_scaled_features(point_features, scale_gate, scale)
        sampled = normed_point_features[torch.rand(normed_point_features.shape[0]) > 0.98]
        sampled = sampled / torch.norm(sampled, dim=-1, keepdim=True)
        
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=epsilon)
        clusterer = hdbscan.HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        )
        cluster_labels = clusterer.fit_predict(sampled.detach().cpu().numpy())
        unique_labels = np.unique(cluster_labels)
        print(len(unique_labels))
        if len(unique_labels) <= 1:
            print(f"[scale={scale:.1f}] Too few clusters: {unique_labels}")
            continue
        
        cluster_centers = torch.zeros(len(unique_labels) - 1, sampled.shape[-1])
        for i in range(1, len(unique_labels)):
            cluster_centers[i - 1] = torch.nn.functional.normalize(
                sampled[cluster_labels == i - 1].mean(dim=0), dim=-1 
            )
            
        seg_score = torch.einsum("nc, bc->bn", cluster_centers.cpu(), normed_point_features.cpu())
        point_label = seg_score.argmax(dim=-1)
        point_label[seg_score.max(dim=-1)[0].detach().cpu().numpy() < similarity_threshold] = -1
        point_label = knn_filled_unlabeled(point_xyz, point_label, k=16)
        np.save(os.path.join(parts_tree_path, f"labels_scale_{scale:.2f}.npy"), point_label.cpu().numpy())
        parts_tree[scale] = point_label
        print(f"[scale={scale:.1f}] {len(unique_labels) - 1} clusters")
    
    return parts_tree

node_id_counter = [0]

def hierarchical_decomposition(
    point_features: torch.Tensor,
    point_xyz: torch.Tensor,
    scale_gate: torch.nn.Module,
    indices: torch.Tensor,
    index_save_root: str,
    current_scale: float = 1.0,
    min_scale: float = 0.0,
    scale_step: float  = 0.2,
    min_cluster_size: int = 10,
    min_samples: int      = 10,
    similarity_threshold: float = 0.5,
    parent_id: Optional[int] = None
):

    makedirs(index_save_root, exist_ok=True)

    node_id = node_id_counter[0]
    node_id_counter[0] += 1

    feats = point_features[indices]
    xyz   = point_xyz[indices]

    # ---------- Transform features by scale ----------
    normed = get_scaled_features(feats, scale_gate, current_scale)
    sampled = normed[torch.rand_like(normed[:, 0]) > 0.98]
    sampled = sampled / torch.norm(sampled, dim=-1, keepdim=True)

    # ---------- HDBSCAN ----------
    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=0.01,
        min_cluster_size=min_cluster_size,
        # min_samples=min_samples,
        allow_single_cluster=True,
        # metric='cosine',
    )
    cluster_labels     = clusterer.fit_predict(sampled.cpu().numpy())
    unique_labels      = np.unique(cluster_labels)
    num_valid_clusters = np.sum(unique_labels >= 0)

    print(f"[Node {node_id}] scale={current_scale:.2f}, pts={len(indices)}, valid={num_valid_clusters}")

    # =============== Case A: Multiple clusters found ===============
    if num_valid_clusters > 1:
        # ---------- Compute labels ----------
        centers = torch.zeros(len(unique_labels) - 1, sampled.shape[-1], device=feats.device)
        for i in range(1, len(unique_labels)):
            centers[i-1] = torch.nn.functional.normalize(
                sampled[cluster_labels == i-1].mean(dim=0), dim=-1
            )
        score = torch.einsum("nc,bc->bn", centers.cpu(), normed.cpu())
        point_label = score.argmax(dim=-1)
        point_label[score.max(dim=-1)[0].detach().cpu().numpy() < similarity_threshold] = -1
        point_label = knn_filled_unlabeled(xyz, point_label)

        children = []
        for cid in point_label.unique():
            assert cid >= 0
            sub_idx = indices[point_label == cid]

            # ---------- If scale has reached 0.0, directly create leaf node ----------
            if current_scale <= min_scale:
                child_id   = node_id_counter[0]; node_id_counter[0] += 1
                fname_child = f"node_{child_id:06d}_scale_{current_scale:.2f}.npy"
                np.save(os.path.join(index_save_root, fname_child), sub_idx.cpu().numpy())

                children.append({
                    "node_id": child_id,
                    "parent_id": node_id,
                    "scale": current_scale,
                    "point_indices_file": fname_child,
                    "children": [],
                    "status": "leaf"
                })
            else:
                # ---------- Not yet at 0.0, continue recursion ----------
                child = hierarchical_decomposition(
                    point_features, point_xyz, scale_gate,
                    indices=sub_idx,
                    index_save_root=index_save_root,
                    current_scale=round(current_scale - scale_step, 2),
                    scale_step=scale_step,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    min_scale=min_scale,
                    similarity_threshold=similarity_threshold,
                    parent_id=node_id
                )
                children.append(child)

        # Save current node indices
        fname = f"node_{node_id:06d}_scale_{current_scale:.2f}.npy"
        np.save(os.path.join(index_save_root, fname), indices.cpu().numpy())

        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "scale": current_scale,
            "point_indices_file": fname,
            "children": children
        }

    # =============== Case B: Cannot form clusters ===============
    if num_valid_clusters <= 1:
        fname = f"node_{node_id:06d}_scale_{current_scale:.2f}.npy"
        np.save(os.path.join(index_save_root, fname), indices.cpu().numpy())
        # (1) If not yet at 0.0, continue trying smaller scale
        if current_scale > min_scale:
            child_node =  hierarchical_decomposition(
                point_features, point_xyz, scale_gate,
                indices=indices,
                index_save_root=index_save_root,
                current_scale=round(current_scale - scale_step, 2),
                scale_step=scale_step,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                min_scale=min_scale,
                similarity_threshold=similarity_threshold,
                parent_id=node_id
            )
            return {
                "node_id": node_id,
                "parent_id": parent_id,
                "scale": current_scale,
                "point_indices_file": fname,
                "children": [child_node]
            }
            
        # (2) Already at 0.0, this is a leaf node
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "scale": current_scale,
            "point_indices_file": fname,
            "children": [],
            "status": "leaf"
        }

def save_tree_to_json(tree: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(tree, f, indent=2)
    print(f"Tree saved to: {save_path}")
    
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--target', default='scene', type=str)
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file")
    args = get_combined_args(parser)

    yaml_config = {}
    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)
    print(args.x_rotation)

    # Set module-level paths from parsed args
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
    dataset.need_features = False

    # To obtain mask scales
    dataset.need_masks = True

    scene_gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')
    bg_color = [0 for i in range(FEATURE_DIM)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    all_scales = []
    for cam in scene.getTrainCameras():
        all_scales.append(cam.mask_scales)
        # print(cam.mask_scales.shape)
    all_scales = torch.cat(all_scales)

    upper_bound_scale = all_scales.max().item()

    # quantile transformer
    q_trans, q_trans_ = get_quantile_func(all_scales, 'uniform')
    
    points_xyz = scene_gaussians.get_xyz
    points_features = feature_gaussians.get_point_features
    
    indices = torch.arange(points_features.shape[0], device=points_features.device)
    index_save_root = os.path.join(MODEL_PATH, "parts_tree")

    # Read part tree parameters from YAML config
    pt_cfg = yaml_config.get("part_tree", {})
    pt_current_scale = pt_cfg.get("current_scale", 0.9)
    pt_scale_step = pt_cfg.get("scale_step", 0.4)
    pt_min_scale = pt_cfg.get("min_scale", 0.1)
    print(f"Part tree params: current_scale={pt_current_scale}, scale_step={pt_scale_step}, min_scale={pt_min_scale}")

    # Reference parameters for other scenes:
    # firewood:      current_scale=0.95, scale_step=0.5, min_scale=0.95
    # playground:    current_scale=0.9,  scale_step=0.4, min_scale=0.1
    # kitchen_new:   current_scale=0.9,  scale_step=0.2, min_scale=0.5
    # kitchen_lego:  current_scale=0.9,  scale_step=0.1, min_scale=0.9
    # kitchen final2: current_scale=0.9, scale_step=0.2, min_scale=0.5

    parts_tree = hierarchical_decomposition(points_features, points_xyz, scale_gate, indices, index_save_root,
                                            current_scale=pt_current_scale, scale_step=pt_scale_step, min_scale=pt_min_scale)
    save_tree_to_json(parts_tree, os.path.join(index_save_root, "parts_tree.json"))
    
    
    # load existed part tree
    with open(os.path.join(index_save_root, "parts_tree.json"), "r") as f:
        parts_tree = json.load(f)
    
    views = scene.getTrainCameras()
    render_seg_path = f"{index_save_root}_vis_seg"
    makedirs(render_seg_path, exist_ok=True)
    
    node_index_files = sorted(glob.glob(os.path.join(index_save_root, "node_*.npy")))
    print(f"Found {len(node_index_files)} nodes.")
    
    nodes_look_up = build_node_lookup(parts_tree)
    leaf_nodes = [nodes_look_up[n] for n in nodes_look_up if is_leaf(nodes_look_up[n])]

    num_clusters = len(leaf_nodes)
    flag_labels = torch.zeros((num_clusters))
    label_view_ratios = {i: [] for i in range(num_clusters)}
    label_view_indices = {i: [] for i in range(num_clusters)}
    
    points_label_mulscale = - torch.ones(points_xyz.shape[0], dtype=torch.int, device=points_xyz.device)
    
    render_path = os.path.join(scene.model_path, "renders",  "rendered")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Seg 3D part")):
        # render_indices = torch.where(flag_labels == 0)
        rendering = render_PGSR(view, scene_gaussians, pipeline.extract(args), background, return_plane=False, return_depth_normal=False)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        for i, node in enumerate(leaf_nodes):
            node_filename = node.get("point_indices_file")
            node_file = os.path.join(index_save_root, node_filename)
            node_indices = np.load(node_file)
            mask_seg = torch.zeros(points_xyz.shape[0], dtype=torch.bool, device=points_xyz.device)
            mask_seg[node_indices] = True
            points_label_mulscale[node_indices] = i
            
            points_fg_num = torch.zeros((points_xyz.size()[0], 2), dtype=torch.int32, device=points_xyz.device)
            ones = torch.ones((points_xyz.size()[0], 1), device=points_xyz.device)
            points_h = torch.cat([points_xyz, ones], dim=1) # [N, 4]
            camera_coords = (view.world_view_transform.T @ points_h.T).T # [N, 4]
            depths = camera_coords[:, 2:3] 
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
            depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points_xyz.device)
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
            visible_mask_all = torch.zeros(points_xyz.size()[0], dtype=torch.bool, device=points_xyz.device)  # [N]
            visible_mask_all[mask_valid] = visible_mask
            mask_depth_label_valid = visible_mask_all & mask_seg
            ratio = (mask_depth_label_valid.sum() / mask_seg.sum()).item()
            label_view_ratios[i].append(ratio)
            label_view_indices[i].append(idx)
            
    assert (points_label_mulscale != -1).all(), "Tensor contains -1 values!"
    np.save(os.path.join(scene.model_path, "points_label_mulscale.npy"), points_label_mulscale.cpu().numpy())
            
    flag_labels = torch.zeros((num_clusters))
    gpt_input_path_sin = os.path.join(scene.model_path, "gpt_input", "gpt_input_single")
    gpt_input_path_mul = os.path.join(scene.model_path, "gpt_input",  "gpt_input_mul")
    render_path = os.path.join(scene.model_path, "renders",  "rendered")
    makedirs(render_path, exist_ok=True)    
    makedirs(gpt_input_path_sin, exist_ok=True)
    makedirs(gpt_input_path_mul, exist_ok=True)

    for i, child_node in enumerate(tqdm(leaf_nodes, desc="Render Seg Parts")):
        child_node_filename = child_node.get("point_indices_file")
        child_node_file = os.path.join(index_save_root, child_node_filename)
        child_node_indices = np.load(child_node_file)
        
        mask_seg = torch.zeros(points_xyz.shape[0], dtype=torch.bool, device=points_xyz.device)
        mask_seg[child_node_indices] = True
            
        max_idx = torch.tensor(label_view_ratios[i]).argmax().item()
        best_view_idx = label_view_indices[i][max_idx]
        
        # ratios_tensor = torch.tensor(label_view_ratios[i])
        # topk_vals, topk_idxs = torch.topk(ratios_tensor, k=2)
        # second_max_idx = topk_idxs[1].item()
        # second_best_view_idx = label_view_indices[i][second_max_idx]
        best_view = views[best_view_idx]
        # best_view = views[second_best_view_idx]
            
        scene_gaussians.segment(mask_seg)
        render_result = render_PGSR(best_view, scene_gaussians, pipeline.extract(args), background, return_plane=True, return_depth_normal=False)
        rendering_seg = render_result["render"]
        rendering_seg_depth = render_result["plane_depth"]
        
        basename = os.path.basename(node_file).replace(".npy", "")
        child_node_id = child_node.get("node_id")
        torchvision.utils.save_image(rendering_seg, os.path.join(render_seg_path, f"{child_node_id}_node_{best_view_idx}_view.png"))
        scene_gaussians.roll_back()
        
        leaf_id = child_node.get("node_id")
        parent_id = child_node.get("parent_id")
        assert parent_id != None and parent_id != "null"
        parent_node = nodes_look_up[parent_id]
        siblings = parent_node.get("children", [])
        
        if len(siblings) == 1 or parent_id == 0:
            # continue
            # Original image
            orig_img_path = os.path.join(render_path, '{0:05d}.png'.format(best_view_idx))
            orig_img = Image.open(orig_img_path).convert("RGB")
            orig_tensor = TF.to_tensor(orig_img)
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(best_view_idx) + ".npy")
            orig_depth = torch.tensor(np.load(depth_path), dtype=torch.float32, device=rendering_seg_depth.device)

            # Create mask (based on segment rendering alpha channel or non-black regions)
            mask_vis = rendering_seg.mean(0) > 0.05
            from scipy import ndimage
            mask_vis_np = mask_vis.cpu().numpy().astype(np.uint8)
            labeled, num_features = ndimage.label(mask_vis_np)

            if num_features > 0:
                # Compute area of each label
                if i == -1:
                    sizes = ndimage.sum(mask_vis_np, labeled, range(1, num_features + 1))
                    sorted_indices = np.argsort(sizes)  # Sort ascending
                    second_largest_index = sorted_indices[-2]  # Second to last (second largest)
                    second_label = second_largest_index + 1    # Labels start from 1
                    largest_mask = (labeled == second_label)
                else:
                    sizes = ndimage.sum(mask_vis_np, labeled, range(1, num_features + 1))
                    max_label = (np.argmax(sizes) + 1)  # Labels start from 1
                    largest_mask = (labeled == max_label)
            else:
                largest_mask = np.zeros_like(mask_vis_np, dtype=bool)

            mask_vis_filtered = torch.from_numpy(largest_mask).to(mask_vis.device)

            mask_depth = (0.1 + orig_depth.squeeze() )>= rendering_seg_depth.squeeze()
            mask =(mask_vis_filtered & mask_depth).cpu().numpy()  # shape: (H, W)

            # Mask overlay image
            overlay = orig_tensor.clone()
            overlay[0][mask] = 0.2  # Blue-ish
            overlay[1][mask] = 0.4
            overlay[2][mask] = 1.0
            overlay = overlay * 0.6 + orig_tensor * 0.4  # Blend original image and blue

            # Bbox image: only show mask region
            ys, xs = torch.where(mask_vis_filtered)
            if ys.numel() == 0:
                part_tensor = torch.zeros_like(orig_tensor)
            else:
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()

                # Optional padding
                pad = 0
                y_min = max(y_min - pad, 0)
                y_max = min(y_max + pad, H - 1)
                x_min = max(x_min - pad, 0)
                x_max = min(x_max + pad, W - 1)

                # Crop bbox region
                crop = rendering_seg[:, y_min:y_max+1, x_min:x_max+1]  # (3, h, w)
                c, h, w = crop.shape

                # Compute target size with aspect ratio preserved
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
            gap_height = 50
            H, W = orig_tensor.shape[1:]

            # Create gap
            gap = torch.ones((3, H, gap_width), dtype=orig_tensor.dtype, device=orig_tensor.device)  # White spacing

            # Insert gap when concatenating
            grid = torch.cat([gap, orig_tensor, gap, overlay, gap, part_tensor, gap], dim=2)
            gap_up = torch.ones((3, gap_height, grid.shape[-1]),  dtype=orig_tensor.dtype, device=orig_tensor.device)
            grid = torch.cat([gap_up, grid], dim=1)

            # Convert to PIL image
            grid_pil = TF.to_pil_image(grid)

            # Add text
            overlay_x0 = gap_width + W + gap_width   # X start of second column (overlay) in padded image
            overlay_y0 = gap_height                           # Y start is always 0

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

            # Extend image height with white background for text (optional)
            # Paste grid_pil onto white background padded image
            padded = Image.new("RGB", (grid_pil.width, grid_pil.height + 50), (255, 255, 255))
            padded.paste(grid_pil, (0, 0))

            # Draw text on padded image
            draw = ImageDraw.Draw(padded)
            labels = ["Original Image", "Mask Overlay", f"Part Image {i}"]
            x_offsets = [gap_width, W + 2*gap_width, 2 * (W + 2*gap_width)]

            for j in range(3):
                text = labels[j]
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                x = x_offsets[j] + (W - text_width) // 2
                y = H + 4 + gap_height
                draw.text((x, y), text, fill="black", font=font)


            # Save
            padded.save(os.path.join(gpt_input_path_sin, '{0:03d}.png'.format(i)))
        
        else:
            # 1. Original image
            # if i != 206 and i!=208:
            #     continue
            orig_img_path = os.path.join(render_path, '{0:05d}.png'.format(best_view_idx))
            orig_img = Image.open(orig_img_path).convert("RGB")
            orig_tensor = TF.to_tensor(orig_img)
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(best_view_idx) + ".npy")
            orig_depth = torch.tensor(np.load(depth_path), dtype=torch.float32, device=rendering_seg_depth.device)
            
            # 2. Parent part
            parent_node_filename = parent_node.get("point_indices_file")
            parent_node_file = os.path.join(index_save_root, parent_node_filename)
            parent_node_indices = np.load(parent_node_file)
            
            mask_seg_parent = torch.zeros(points_xyz.shape[0], dtype=torch.bool, device=points_xyz.device)
            mask_seg_parent[parent_node_indices] = True
                
            scene_gaussians.segment(mask_seg_parent)
            render_result_parent = render_PGSR(best_view, scene_gaussians, pipeline.extract(args), background, return_plane=True, return_depth_normal=False)
            rendered_parent, render_depth_parent = render_result_parent['render'], render_result_parent["plane_depth"]
            scene_gaussians.roll_back()
            
            origin_mask_depth = (0.1 + orig_depth.squeeze() )>= render_depth_parent.squeeze()

            
            # 3. Overlay
            # Create mask (based on segment rendering alpha channel or non-black regions)
            mask_vis = rendering_seg.mean(0) > 0.05
            mask_vis_origin = rendered_parent.mean(0) > 0.05
            from scipy import ndimage
            mask_vis_np = mask_vis.cpu().numpy().astype(np.uint8)
            labeled, num_features = ndimage.label(mask_vis_np)
            
            origin_mask = (origin_mask_depth & mask_vis_origin).cpu().numpy()
            origin_overlay = orig_tensor.clone()
            origin_overlay[0][origin_mask] = 1.0  # Blue-ish
            origin_overlay[1][origin_mask] = 1.0
            origin_overlay[2][origin_mask] = 0.0
            origin_overlay = origin_overlay * 0.6 + orig_tensor * 0.4

            if num_features > 0:
                # Compute area of each label
                sizes = ndimage.sum(mask_vis_np, labeled, range(1, num_features + 1))
                max_label = (np.argmax(sizes) + 1)  # Labels start from 1
                largest_mask = (labeled == max_label)
            else:
                largest_mask = np.zeros_like(mask_vis_np, dtype=bool)

            mask_vis_filtered = torch.from_numpy(largest_mask).to(mask_vis.device)

            mask_depth = (0.1 + orig_depth.squeeze() )>= rendering_seg_depth.squeeze()
            # mask =(mask_vis_filtered & mask_depth).cpu().numpy()  # shape: (H, W)
            mask = mask_vis_filtered.cpu().numpy()

            # Mask overlay image
            overlay = rendered_parent.clone()
            overlay[0][mask] = 0.2  # Blue-ish
            overlay[1][mask] = 0.4
            overlay[2][mask] = 1.0
            overlay = overlay * 0.6 + rendered_parent * 0.4  # Blend original image and blue
            
            # 4. Part
            ys, xs = torch.where(mask_vis_filtered)
            if ys.numel() == 0:
                part_tensor = torch.zeros_like(orig_tensor)
            else:
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()

                # Optional padding
                pad = 0
                y_min = max(y_min - pad, 0)
                y_max = min(y_max + pad, H - 1)
                x_min = max(x_min - pad, 0)
                x_max = min(x_max + pad, W - 1)

                # Crop bbox region
                crop = rendering_seg[:, y_min:y_max+1, x_min:x_max+1]  # (3, h, w)
                c, h, w = crop.shape

                # Compute target size with aspect ratio preserved
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

            # Construct four-panel image
            gap_width = 20
            gap_height = 50
            H, W = orig_tensor.shape[1:]

            # Create gap
            gap = torch.ones((3, H, gap_width), dtype=orig_tensor.dtype, device=orig_tensor.device)  # White spacing


            # Insert gap when concatenating
            grid = torch.cat([gap, origin_overlay, gap, rendered_parent.cpu(), gap, overlay.cpu(), gap, part_tensor, gap], dim=2)
            gap_up = torch.ones((3, gap_height, grid.shape[-1]),  dtype=orig_tensor.dtype, device=orig_tensor.device)
            grid = torch.cat([gap_up, grid], dim=1)

            # Convert to PIL image
            grid_pil = TF.to_pil_image(grid)

            # Add text
            overlay_x0 = 3 * gap_width + 2 * W  # X start of third column (overlay) in padded image
            overlay_y0 = gap_height                           # Y start is always 0

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

            # Extend image height with white background for text (optional)
            # Paste grid_pil onto white background padded image
            padded = Image.new("RGB", (grid_pil.width, grid_pil.height + 50), (255, 255, 255))
            padded.paste(grid_pil, (0, 0))

            # Draw text on padded image
            draw = ImageDraw.Draw(padded)
            labels = ["Original Image", "Parent Part", "Overlay", f"Part Image {i}"]
            x_offsets = [gap_width, W + 2*gap_width, 2 * (W + 2*gap_width), 3 * (W + 2*gap_width)]

            for j in range(4):
                text = labels[j]
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                x = x_offsets[j] + (W - text_width) // 2
                y = H + 4 + gap_height
                draw.text((x, y), text, fill="black", font=font)


            # Save
            padded.save(os.path.join(gpt_input_path_mul, '{0:03d}.png'.format(i)))
            
            
        try:
            scene_gaussians.roll_back()
        except:
            pass

        
