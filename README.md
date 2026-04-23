# FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting (ICLR 2026)

### [Paper](https://openreview.net/forum?id=ziKFH7whvy) | [Project Page](https://pku-vcl-geometry.github.io/FieryGS/)


<!-- TODO: Add teaser figure -->
<!-- ![teaser](assets/teaser.png) -->

## Abstract
We consider the problem of synthesizing photorealistic, physically plausible combustion effects in in-the-wild 3D scenes. Traditional CFD and graphics pipelines can produce realistic fire effects but rely on handcrafted geometry, expert-tuned parameters, and labor-intensive workflows, limiting their scalability to the real world. Recent scene modeling advances like 3D Gaussian Splatting (3DGS) enable high-fidelity real-world scene reconstruction, yet lack physical grounding for combustion. To bridge this gap, we propose FieryGS, a physically-based framework that integrates physically-accurate and user-controllable combustion simulation and rendering within the 3DGS pipeline, enabling realistic fire synthesis for real scenes. Our approach tightly couples three key modules: (1) multimodal large-language-model-based physical material reasoning, (2) efficient volumetric combustion simulation, and (3) a unified renderer for fire and 3DGS. By unifying reconstruction, physical reasoning, simulation, and rendering, FieryGS removes manual tuning and automatically generates realistic, controllable fire dynamics consistent with scene geometry and materials. Our framework supports complex combustion phenomena—including flame propagation, smoke dispersion, and surface carbonization—with precise user control over fire intensity, airflow, ignition location and other combustion parameters. Evaluated on diverse indoor and outdoor scenes, FieryGS outperforms all comparative baselines in visual realism, physical fidelity, and controllability.

## News
- **[2026-04]** Code released.
- **[2026-02]** Paper accepted at ICLR 2026.

---

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:qianfanshen/FireGaussian.git
cd FireGaussian
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate fire_gaussian
```

Required environment:
- Python 3.9
- CUDA 11.8
- PyTorch 2.5.1

### 3. Install submodules and dependencies
After activating the conda environment, install the submodules and additional dependencies:
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization_contrastive_f
pip install submodules/diff-gaussian-rasterization-depth
pip install submodules/diff-plane-rasterization
pip install submodules/diff-plane-rasterization-contrastive-f
pip install submodules/simple-knn
pip install third_party/segment-anything
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 4. Download segmentation checkpoints
```bash
mkdir -p checkpoints
# SAM (default)
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# SAM2 (alternative, requires hydra-core)
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 5. Set up environment variables
Copy the example config and fill in your API keys (required for VLM material prediction):
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and QWEN_API_KEY
source .env
```

---

## Quick Start

To run the full pipeline on a scene in one command:
```bash
python run_MR.py --scenes firewoods_sand_dark --factors 2 --gpu_id 0
```

This runs all 10 steps automatically. You can also run each step individually as described below.

---

## Step-by-Step Pipeline

### Step 1: Train 3D Gaussians (PGSR)
```bash
CUDA_VISIBLE_DEVICES=0 python train_scene_PGSR.py \
    -s data/<scene> \
    -m output/<scene> \
    -r <downsample: 1/2/4/8> \
    --config_file ./arguments/<scene>.yaml \
    --densify_abs_grad_threshold 0.0002
```

### Step 2: Extract Segmentation Masks
Using SAM (default):
```bash
CUDA_VISIBLE_DEVICES=0 python extract_sam2_mask.py \
    --image_root data/<scene> \
    -s sam \
    --downsample <1/2/4/8>
```
Or using SAM2 (alternative):
```bash
CUDA_VISIBLE_DEVICES=0 python extract_sam2_mask.py \
    --image_root data/<scene> \
    -s sam2 \
    --downsample <1/2/4/8>
```

### Step 3: Compute Mask Scales
```bash
CUDA_VISIBLE_DEVICES=0 python get_scale_PGSR.py \
    --image_root data/<scene> \
    --model_path output/<scene> \
    --config_file ./arguments/<scene>.yaml
```

### Step 4: Train 3D Gaussian Affinity Features
```bash
CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature_PGSR.py \
    -m output/<scene> \
    --iterations 10000 \
    --num_sampled_rays 1000 \
    --config_file ./arguments/<scene>.yaml
```

### Step 5: Render Scene
```bash
CUDA_VISIBLE_DEVICES=0 python render_scene_PGSR.py \
    -m output/<scene> \
    --config_file ./arguments/<scene>.yaml
```

### Step 6: Extract Hierarchical Part Tree
```bash
CUDA_VISIBLE_DEVICES=0 python extract_part_tree.py \
    --config_file ./arguments/<scene>.yaml
```

### Step 7: VLM Material Prediction
Requires `OPENAI_API_KEY` or `QWEN_API_KEY` environment variable.
```bash
CUDA_VISIBLE_DEVICES=0 python vlm_predict.py \
    --dataset_path output/<scene>/gpt_input \
    --vlm gpttool
```

### Step 8: Visualize Material Segmentation
```bash
CUDA_VISIBLE_DEVICES=0 python vis_material_seg.py \
    --config_file ./arguments/<scene>.yaml
```

### Step 9: Extract Voxels for Fire Simulation
```bash
CUDA_VISIBLE_DEVICES=0 python extract_voxel.py \
    -m output/<scene> \
    --config_file ./arguments/<scene>.yaml
```

The voxel attributes for fire simulation are saved at `output/<scene>/simulation_voxel.yaml`.

### Step 10: Consolidate Outputs to Data Directory
```bash
python consolidate_to_data.py \
    -s data/<scene> \
    -m output/<scene> \
    --config_file ./arguments/<scene>.yaml
```

Copies key outputs (point cloud, voxel grids, depth maps, etc.) from `output/<scene>` into `data/<scene>` so that all assets needed for fire simulation and rendering are in one place.

---

## Fire Simulation & Rendering

After obtaining the voxel attributes, run the fire simulation:
```bash
CUDA_VISIBLE_DEVICES=0 python -m simulation.fire_main --config_file arguments/<scene>.yaml
```
The simulation result can be found at `output/<scene>/sim_output`.

Render with fire effects:
```bash
CUDA_VISIBLE_DEVICES=0 python -m rendering.scene_render_fire -m data/<scene> --config_file arguments/<scene>.yaml
```
The rendering result can be found at `output/<scene>/render_output`.

---

## Scene Configuration

Scene-specific parameters are defined in `./arguments/<scene>.yaml`:

| Field | Description |
|-------|-------------|
| `dataset_name` | Scene name |
| `x_rotation` | Camera rotation offset for rendering |
| `fire_sim_root` | Path to fire simulation output |
| `is_render_360` | Whether to render a 360-degree camera path instead of dataset views |
| `extract_voxel` | Initial tight voxel extraction box (`bbox_min`, `bbox_max`) and extraction resolution/scale |
| `bounding_box` | Scene AABB for voxelization |
| `voxel_grid` | Voxel grid dimensions and voxel size |
| `load_path*` | Cached voxel/grid index masks used to initialize simulation faster |
| `sim_frames` | Number of fire simulation frames to run |
| `render_smoke` | Enable smoke rendering contribution |
| `render_single` | Render only one frame/camera (`render_frame`, `render_camera`) for debugging |
| `camera_params` | Custom orbit camera settings for 360 rendering (radius, azimuth, elevation, offsets, look-up point) |

---

##  Data

<!-- TODO: Update download links for release scenes -->

| Scene | Data |
|-------|------|
| Firewoods_sand_dark | [Google Drive](https://drive.google.com/file/d/1EMWrcSLbeLjTWyGUh_cThYGvrrmMrvC0/view?usp=sharing) |
| Playground | [Google Drive](https://drive.google.com/file/d/1NFXzr0YU3q1_ezjWVEgTmd-R8JX4J0lL/view?usp=sharing) |

**Expected directory structure:**
```
FireGaussian/
├── data/
│   └── <scene>/
│       ├── images/              # Input images
│       ├── sparse/              # COLMAP sparse reconstruction
│       ├── normals/             # (optional)
│       ├── 360/                 # 360° renders (after Step 10)
│       ├── app_model/           # Appearance model
│       ├── fg/                  # Foreground (burnable) Gaussians
│       ├── point_cloud/         # Gaussian point clouds & features
│       ├── depth/               # Depth maps
│       ├── renders_depth/       # Rendered depth images
│       ├── sim_output/          # Fire simulation output
│       ├── cfg_args             # Training config
│       ├── cameras.json         # Camera parameters
│       ├── input.ply            # Input point cloud
│       ├── voxel_grid.npy       # Voxel occupancy grid
│       ├── material_grid.npy    # Material voxel grid
│       ├── mask_pts_in_grids.pth
│       └── indices_pts_in_grids.pth
└── output/
    └── <scene>/                 # Full pipeline outputs
        ├── sim_output/          # Fire simulation output
        └── render_output/       # Rendering output
```

---

## Acknowledgements

This project builds upon several excellent works:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [PGSR](https://github.com/zju3dv/PGSR)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [SAGA](https://github.com/Jumpat/SegAnyGAussians)

## License

This project is licensed under the terms in [LICENSE.md](LICENSE.md).

## Citation

```bibtex
@InProceedings{shen2026fierygs,
  title = {FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting},
  author = {Qianfan Shen and Ningxiao Tao and Qiyu Dai and Tianle Chen and Minghan Qin and Yongjie Zhang and Mengyu Chu and Wenzheng Chen and Baoquan Chen},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year = {2026},
  url = {https://openreview.net/forum?id=ziKFH7whvy}
}
```
