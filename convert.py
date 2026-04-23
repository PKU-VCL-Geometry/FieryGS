# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

# For inquiries contact  george.drettakis@inria.fr


import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# from PIL import Image

# # os.makedirs(args.source_path + "/input", exist_ok=True) 
# # files = os.listdir(args.source_path + "/input_")
# # files = [f for f in os.listdir(args.source_path + "/input_") if not f.startswith('.')]


# # for file in files:
# #     source_file = os.path.join(args.source_path, "input_", file)
# #     destination_file = os.path.join(args.source_path, "input", file)

# #     try:
# #         # Open the original image
# #         img = Image.open(source_file)
# #         # Calculate new dimensions
# #         new_size = (img.width // 2, img.height // 2)
# #         # Resize and save
# #         img_resized = img.resize(new_size, Image.LANCZOS)  # Use high-quality interpolation
# #         img_resized.save(destination_file)

# #     except Exception as e:
# #         logging.error(f"Resize failed for {file} due to: {e}")
# #         exit(1)

        
if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = (
        colmap_command
        + " feature_extractor "
        + "--database_path " + args.source_path + "/distorted/database.db "
        + "--image_path " + args.source_path + "/input "
        + "--ImageReader.single_camera 1 "
        + "--ImageReader.camera_model " + args.camera + " "
        + "--SiftExtraction.use_gpu " + str(use_gpu)
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching (using sequential mode)
    feat_matching_cmd = (
        colmap_command
        + " sequential_matcher "
        + "--database_path " + args.source_path + "/distorted/database.db "
        + "--SiftMatching.use_gpu " + str(use_gpu) + " "
        + "--SequentialMatching.overlap 10 "
        + "--SequentialMatching.loop_detection false "
        # If you need to sort by filename, you can specify "--image_list_path".
        # For more parameters, refer to the official documentation or help: colmap sequential_matcher --help
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (
        colmap_command
        + " mapper "
        + "--database_path " + args.source_path + "/distorted/database.db "
        + "--image_path " + args.source_path + "/input "
        + "--output_path " + args.source_path + "/distorted/sparse "
        + "--Mapper.ba_global_function_tolerance=0.000001"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (
    colmap_command
    + " image_undistorter "
    + "--image_path " + args.source_path + "/input "
    + "--input_path " + args.source_path + "/distorted/sparse/0 "
    + "--output_path " + args.source_path + " "
    + "--output_type COLMAP"
)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    files = os.listdir(args.source_path + "/images")
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
# print("Running model_orientation_aligner...")
# aligned_output_path = os.path.join(args.source_path, "sparse_aligned")
# os.makedirs(aligned_output_path, exist_ok=True)           
# orient_aligner_cmd = (
#     colmap_command
#     + " model_orientation_aligner "
#     + "--image_path " + os.path.join(args.source_path, "images") + " "
#     + "--input_path " + os.path.join(args.source_path, "sparse", "0") + " "
#     + "--output_path " + aligned_output_path
# )
# exit_code = os.system(orient_aligner_cmd)
# if exit_code != 0:
#     logging.error(f"Orientation alignment failed with code {exit_code}. Exiting.")
#     exit(exit_code)

print("Done.")
