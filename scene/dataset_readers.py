#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from tqdm import tqdm
from tqdm import trange
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation as R

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # treehill: keep this part, comment out the rest of the alignment
    ##########################################################################
    # Flip coordinate system if z component of y-axis is negative
    # if poses_recentered.mean(axis=0)[2, 1] < 0:
    #     poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    #     transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    ##########################################################################

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = 1.0

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_poses_and_transform(poses, transform, rotation_angle_x_deg=0, rotation_angle_y_deg=0):
    """Applies rotations around the X and Y axes to the poses and transform.

    Args:
      poses: A (N, 3, 4) array of camera poses.
      transform: A 4x4 matrix representing the global transformation.
      rotation_angle_x_deg: Rotation angle in degrees around the X axis.
      rotation_angle_y_deg: Rotation angle in degrees around the Y axis.

    Returns:
      A tuple (rotated_poses, rotated_transform) with the updated poses and transform.
    """
    # 1. Create rotation matrices around the X and Y axes
    rotation_angle_x_rad = np.deg2rad(rotation_angle_x_deg)  # Convert degrees to radians
    rotation_angle_y_rad = np.deg2rad(rotation_angle_y_deg)

    # X-axis rotation matrix
    rotation_x = R.from_euler('x', rotation_angle_x_rad).as_matrix()

    # Y-axis rotation matrix
    rotation_y = R.from_euler('y', rotation_angle_y_rad).as_matrix()

    # 2. Create 4x4 rotation matrices to apply to the transform matrix
    rotation_x_4x4 = np.eye(4)
    rotation_x_4x4[:3, :3] = rotation_x  # Insert rotation matrix into the rotation part only

    rotation_y_4x4 = np.eye(4)
    rotation_y_4x4[:3, :3] = rotation_y  # Insert rotation matrix into the rotation part only

    # 3. Apply rotation matrices to all poses and transform
    rotated_poses = np.copy(poses)  # Copy original poses
    rotated_transform = np.copy(transform)  # Copy original transform

    # Apply rotation to each pose
    for i in range(poses.shape[0]):
        rotated_poses[i, :3, 3] = rotation_y @ rotation_x @ poses[i, :3, 3]  # Rotate translation part: Y-axis first, then X-axis
        rotated_poses[i, :3, :3] = rotation_y @ rotation_x @ poses[i, :3, :3]  # Rotate rotation matrix part: Y-axis first, then X-axis

    # Apply rotation to the overall transform
    rotated_transform[:3, :3] = rotation_y @ rotation_x @ transform[:3, :3]  # Rotate transform's rotation part: Y-axis first, then X-axis
    rotated_transform[:3, 3] = rotation_y @ rotation_x @ transform[:3, 3]  # Rotate transform's translation part: Y-axis first, then X-axis

    return rotated_poses, rotated_transform

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    normal_image: np.array
    features: torch.tensor
    masks: torch.tensor
    mask_scales: torch.tensor
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float = None
    cy: float = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, features_folder = None, masks_folder = None, mask_scale_folder = None, sample_rate = 1.0, allow_principle_point_shift = False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        if idx % 10 >= sample_rate * 10:
            continue
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # print(intr.model)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length = intr.params[0]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
        else:
            assert False, f"Colmap camera model {intr.model} not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
             
        sky_path = image_path.replace("images", "sky_mask")[:-4]+".npy"
        if os.path.exists(sky_path):
            sky_mask = np.load(sky_path).astype(np.uint8)
        else:
            # print("No sky mask!")
            sky_mask = None
            
        floor_path = image_path.replace("images", "floor_mask")[:-4]+".npy"
        if os.path.exists(floor_path):
            floor_mask = np.load(floor_path).astype(np.uint8)
        else:
            # print("No floor mask!")
            floor_mask = None
            
        normal_path = image_path.replace("images", "normals")[:-4]+".npz"
        if os.path.exists(normal_path):
            normal_image = np.load(normal_path)['arr_0']
        else:
            normal_image = None
            # The above code is a Python script that prints the message "No floor mask!" to the
            # console.
            print("No Normal Prior!")
            
        image = Image.open(image_path)
        # print(features_folder)
        features = torch.load(os.path.join(features_folder, image_name.split('.')[0] + ".pt")) if features_folder is not None else None
        # print(features)
        masks = torch.load(os.path.join(masks_folder, image_name.split('.')[0] + ".pt")) if masks_folder is not None else None
        mask_scales = torch.load(os.path.join(mask_scale_folder, image_name.split('.')[0] + ".pt")) if mask_scale_folder is not None else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, features=features, masks=masks, mask_scales = mask_scales, normal_image=normal_image,
                              image_path=image_path, image_name=image_name, width=width, height=height, cx=intr.params[2] if len(intr.params) > 3 and allow_principle_point_shift else None, cy=intr.params[3] if len(intr.params) >3 and allow_principle_point_shift else None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, only_xyz=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors, normals = None, None
    if not only_xyz:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, need_features=False, need_masks=False, sample_rate = 1.0, allow_principle_point_shift = False, replica=False, x_rotation=0, y_rotation=0):
        # try:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    feature_dir = "clip_features"
    mask_dir = "sam_masks" if os.path.exists(os.path.join(path, "sam_masks")) else "sam2_masks"
    mask_scale_dir = "mask_scales_PGSR"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), features_folder=os.path.join(path, feature_dir) if need_features else None, masks_folder=os.path.join(path, mask_dir) if need_masks else None, mask_scale_folder=os.path.join(path, mask_scale_dir) if need_masks else None, sample_rate=sample_rate, allow_principle_point_shift = allow_principle_point_shift)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    # Without alignment
    #############################################################################################################
    # if train_list is not None:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
    #     print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    # elif eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = []

    # nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "sparse/0/points3D_colmap_aligned.ply")
    # bin_path = os.path.join(path, "sparse_aligned/points3D.bin")
    # txt_path = os.path.join(path, "sparse_aligned/points3D.txt")
    # if not os.path.exists(ply_path) or True:
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #         print(f"xyz {xyz.shape}")
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    # scene_info = SceneInfo(point_cloud=pcd,
    #                        train_cameras=train_cam_infos,
    #                        test_cameras=test_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path)
    #############################################################################################################
    
    # align
    #############################################################################################################
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    # if not os.path.exists(ply_path):
        # print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

        
        
    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws)
    # print(transform)
    # garden
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, 187.5)
    # firewood_sand
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, 172)
    # firewoods_sand
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, -5)
    # chair
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, -9)
    
    print(f"x_rotation: {x_rotation}, y_rotation: {y_rotation}")
    # kitchen
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, 15)
    #kitchen_lego
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, 5)
    #kitchen_final
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, 6)
    #kitchen_final2
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, -4)
    # kitchen_iclr
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, -7)
    # kitchen_iclr2
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, 5)
    # chair_mul
    c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation, 4)
    # others
    # c2ws, transform = rotate_poses_and_transform(c2ws, transform, x_rotation)
    print(transform)
    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        R_transpose = np.transpose(R)
    
        
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Align point cloud
    if pcd is not None:
        pointcloud = np.array(pcd.points) 
        pointcloud_aligned = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
        rotation_matrix = transform[:3, :3]  # Extract the 3x3 rotation matrix
        t = transform[:3, 3]    # Extract the 3x1 translation vector
        print("Translation vector t:")
        print(t)
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)  # Return rotation angles
        print("Rotation angles (along x, y, z axes):")
        print(euler_angles)
        aligned_ply_path = os.path.join(path, "sparse/0/points3D_aligned_PGSR.ply")
        storePly(aligned_ply_path, pointcloud_aligned, rgb)
        pcd = fetchPly(aligned_ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=aligned_ply_path)
    #############################################################################################################
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

# for lerf test
def readCamerasFromLerfTransforms(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]

        

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"])

            tmp = np.array(frame["transform_matrix"])
            tmp_R = tmp[:3,:3]
            tmp_R = -tmp_R
            tmp_R[:,0] = -tmp_R[:,0]
            tmp[:3,:3] = tmp_R
            matrix = np.linalg.inv(tmp)
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # matrix[:3,1] *= -1
            # matrix[:3,2] *= -1

            R = np.transpose(matrix[:3,:3])
            T = matrix[:3, 3]

            image_path = os.path.join(path, frame["file_path"])
            image_name = frame["file_path"].split("/")[-1].split(".")[0]
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            fovx = 2 * np.arctan(frame['w'] / (2 * frame['fl_x']))
            fovy = 2 * np.arctan(frame['h'] / (2 * frame['fl_y']))

            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], features = None, masks = None, mask_scales = None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readLerfInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromLerfTransforms(path, "transforms.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    test_cam_infos = []
    eval = False
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Lerf" : readLerfInfo
}