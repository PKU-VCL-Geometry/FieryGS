import os
import sys
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

# Add Grounded-SAM-2 to path so that 'sam2' package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Grounded-SAM-2"))
# from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
#                               sam_model_registry)
# from segment_anything_hq import (SamAutomaticMaskGenerator, SamPredictor,
#                               sam_model_registry)
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", "-i", required=True, type=str)
    parser.add_argument("--sam_checkpoint_path", default='./checkpoints/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam2_checkpoint_path", default='./checkpoints/sam2.1_hiera_large.pt', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--segmenter", "-s", default='sam', type=str, choices=['sam', 'sam2', 'sam_hq', 'sam2_hq'], help="Use original SAM or SAM2")
    parser.add_argument("--downsample", "-r", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")
    parser.add_argument("--visualize", "-v", action='store_true', help="Visualize and save overlay of masks")

    args = parser.parse_args()
    
    
    print("Initializing SAM...")
    # -----------------------------
    # Setup segmenter
    # -----------------------------
    print(f"Using segmenter: {args.segmenter}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if args.segmenter == 'sam2':
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_ckpt = args.sam2_checkpoint_path
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_ckpt, device=device, apply_postprocessing=False)

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            stability_score_offset=0.7,
            crop_n_layers=0,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            use_m2m=True,
        )
    elif args.segmenter == "sam":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
        sam = sam_model_registry[args.sam_arch](checkpoint=args.sam_checkpoint_path).to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88, #Playground 0.85, others 0.88
            box_nms_thresh=0.7,
            stability_score_thresh=0.95,#Playground 0.9, others 0.95
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
    elif args.segmenter == "sam_hq":
        from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint_path).to(device)
        mask_generator = SamAutomaticMaskGenerator(
                            model=sam,
                            points_per_side=32,
                            pred_iou_thresh=0.75,
                            stability_score_thresh=0.85,
                            crop_n_layers=1,
                            crop_n_points_downscale_factor=2,
                            min_mask_region_area=100,  # Requires open-cv to run post-processing
                        )
    elif args.segmenter == 'sam2_hq':
        from sam2_hq.build_sam import build_sam2
        from sam2_hq.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_ckpt = args.sam2_checkpoint_path
        model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_ckpt, device=device, apply_postprocessing=False)

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            stability_score_offset=0.7,
            crop_n_layers=0,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            use_m2m=True,
        )

    downsample_manually = False
    if args.downsample == "1" or args.downsample_type == 'mask':
        IMAGE_DIR = os.path.join(args.image_root, 'images')
    else:
        IMAGE_DIR = os.path.join(args.image_root, 'images_'+str(args.downsample))
        if not os.path.exists(IMAGE_DIR):
            IMAGE_DIR = os.path.join(args.image_root, 'images')
            downsample_manually = True
            print("No downsampled images, do it manually.")

    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, f'{args.segmenter}_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    VISUALIZE_DIR = os.path.join(args.image_root, f'{args.segmenter}_mask_visualizations')
    os.makedirs(VISUALIZE_DIR, exist_ok=True) 
    
    print("Extracting SAM segment everything masks...")
    
    count = 0
    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        # if count % 5 == 0:

        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if downsample_manually:
            img = cv2.resize(img,dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.resize(img_rgb,dsize=(img_rgb.shape[1] // args.downsample, img_rgb.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        masks = mask_generator.generate(img)
        # print(len(masks))
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if args.downsample_type == 'mask':
                m_score = torch.nn.functional.interpolate(m_score.unsqueeze(0).unsqueeze(0), size=(img.shape[0] // args.downsample, img.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                m_score = m_score.bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        masks = torch.stack(mask_list, dim=0)

        torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))
        # count += 1
        if args.visualize:
            vis_img = img_rgb.copy()
            color_map = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
            for idx, m in enumerate(masks):
                m_np = m.cpu().numpy().astype(np.uint8)
                color = color_map[idx]
                colored_mask = np.stack([m_np * c for c in color], axis=-1)
                vis_img = np.where(m_np[..., None] == 1, 0.2 * vis_img + 0.8 * colored_mask, vis_img)

            vis_img = vis_img.astype(np.uint8)
            Image.fromarray(vis_img).save(os.path.join(VISUALIZE_DIR, name + '_vis.png'))