
<p align="center">
  <!-- You can create and add a logo/banner here for OccVerse -->
  <!-- <img src="path/to/your/banner.png" width="80%"> -->
  <h1 align="center">OccVerse: A Unified Framework for 3D Occupancy Prediction</h1>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8+-orange.svg"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0.1-red.svg"></a>
    <a href="https://github.com/cdb342/OccVerse/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
</p>


Welcome to **OccVerse**, a framework for 3D Occupancy Prediction. This project unifies our previous works, including **ALOcc**, **CausalOcc**, and **GDFusion**, into a single codebase to support research in autonomous driving perception. The framework is designed to handle both **Semantic Occupancy** and **Occupancy Flow** prediction.

Our goal is to provide a flexible and unified tool to accelerate academic research and industry applications. We hope OccVerse can serve as a solid foundation for the community to build upon.


---

## 🌟 Highlights

- 🏆 **A Unified Framework**: Provides a common codebase for our works (**ALOcc**, **CausalOcc**, **GDFusion**) and other models like `BEVDetOcc`, `FB-Occ`, etc.
- 🔧 **Configurable Feature Representation**: Supports multiple type of 3D feature encoding (like Volume-based and BEV-based), switchable via configuration.
- 📚 **Dataset Support**: Provides full support for large-scale datasets like **nuScenes** and **Waymo**, and allows for seamlessly switching between different occupancy annotation formats (e.g., **Occ3D**, **SurroundOcc**, **OpenOccupancy**) for robust experimentation.

---

## 🛠 Model Zoo

OccVerse supports the following state-of-the-art models:

| Method        | Task                      | Publication |
|---------------|---------------------------|-------------|
| ALOcc         | Semantic Occupancy & Flow | ICCV 2025   |
| CausalOcc     | Semantic Occupancy        | ICCV 2025   |
| GDFusion      | Semantic Occupancy        | CVPR 2025   |
| BEVDetOcc     | Semantic Occupancy        | -           |
| FB-Occ        | Semantic Occupancy        | ICCV 2023   |
| BEVFormer     | Semantic Occupancy        | ECCV 2022   |
| SparseOcc     | Semantic Occupancy        | ECCV 2023   |

---

## 🚀 Get Started

### 1. Installation

We recommend using Conda for environment management.

```bash
# Clone this repository (replace OccVerse with your actual repo name)
git clone https://github.com/cdb342/OccVerse
cd OccVerse

# Create and activate the conda environment
conda create -n occverse python=3.8 -y
conda activate occverse

# Install PyTorch dependencies (for CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV dependencies
git clone https://github.com/open-mmlab/mmcv
cd mmcv
git checkout 1.x # Use the stable 1.x branch
MMCV_WITH_OPS=1 pip install -e . -v
cd ..

# Install MMDetection and MMSegmentation
pip install mmdet==2.28.2 mmsegmentation==0.30.0

# Install the OccVerse framework itself
pip install -v -e .

# Install other dependencies
pip install torchmetrics timm dcnv4 ninja spconv transformers IPython einops
pip install numpy==1.23.4 # Pin numpy version for compatibility
```

### 2. Data Preparation

#### **nuScenes**

1.  Download the full nuScenes dataset from the [official website](https://www.nuscenes.org/download).
2.  Download the Occ3D nuScenes annotations from the [project page](https://tsinghua-mars-lab.github.io/Occ3D/).
3.  (Optional) Download other community annotations for extended experiments:
    *   [OpenOcc_v2.1 Annotations](https://github.com/OpenDriveLab/OccNet?tab=readme-ov-file#openocc-dataset)
    *   [OpenOcc_v2.1 Ray Mask](https://drive.google.com/file/d/10jB08Z6MLT3JxkmQfxgPVNq5Fu4lHs_h/view)
    *   [SurroundOcc Annotations](https://github.com/weiyithu/SurroundOcc) (rename to `gts_surroundocc`)
    *   [OpenOccupancy-v0.1 Annotations](https://github.com/JeffWang987/OpenOccupancy)

Please organize the data into the following directory structure:

```
├── data
│   ├── nuscenes
│   │   ├── maps, samples, sweeps, v1.0-test, v1.0-trainval
│   │   ├── gts                 # Occ3D annotations
│   │   ├── gts_surroundocc     # (Optional) SurroundOcc annotations
│   │   ├── openocc_v2          # (Optional) OpenOcc annotations
│   │   ├── openocc_v2_ray_mask # (Optional) OpenOcc ray mask
│   │   └── nuScenes-Occupancy-v0.1 # (Optional) OpenOccupancy annotations
```


Finally, run the preprocessing scripts:


```bash
# Prepare LiDAR segmentation labels
python tools/nusc_process/extract_sem_point.py

# Create formatted data for training
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
python tools/create_data_bevdet.py
```

#### **Waymo**

1. Download the Waymo Open Dataset from the [official website](https://waymo.com/open/download/).
2. Download the Occ3D Waymo annotations and pkl files from [here](https://github.com/Tsinghua-MARS-Lab/CVT-Occ/blob/main/docs/dataset.md).
3. Follow the official instructions to organize the files.

### 3. Pre-trained Models

For training, please download pre-trained image backbones from [BEVDet GitHub](https://github.com/HuangJunJie2017/BEVDet), [GeoMIM GitHub](https://github.com/Sense-X/GeoMIM), or [Hugging Face Hub](https://huggingface.co/Dobbin/alocc). Place them in the `ckpts/pretrain/` directory.


---

## 🎮 Usage

### Training

Use the following script for distributed training.

```bash
# Syntax: bash tools/dist_train.sh [CONFIG_FILE] [WORK_DIR] [NUM_GPUS]
# Example: Train the ALOcc-3D model
bash tools/dist_train.sh configs/alocc/alocc_3d_256x704_bevdet_preatrain.py work_dir/alocc_3d 8
```

### Testing

Download our pre-trained models from [Hugging Face](https://huggingface.co/Dobbin/alocc) and run the testing script.

```bash
# Evaluation with mIoU
# Syntax: bash tools/dist_test.sh [CONFIG_FILE] [CHECKPOINT_PATH] [NUM_GPUS]
# Example: Evaluate the ALOcc-3D model
bash tools/dist_test.sh configs/alocc/alocc_3d_256x704_bevdet_preatrain.py ckpts/alocc_3d_256x704_bevdet_preatrain.pth 8

# Evaluation with RayIoU
# Syntax: bash tools/dist_test_ray.sh [CONFIG_FILE] [CHECKPOINT_PATH] [NUM_GPUS]
# Example: Evaluate the ALOcc-3D model
bash tools/dist_test.sh configs/alocc/alocc_3d_256x704_bevdet_preatrain_wo_mask.py ckpts/alocc_3d_256x704_bevdet_preatrain_wo_mask.pth 8
```

> **Note:** Due to an unresolved bug in the sampler, please use **1 or 8 GPUs** for inference to ensure accurate metric calculation. Using a different number of GPUs may lead to duplicate sample counting.

### Benchmarking

We provide convenient tools to benchmark model FPS (Frames Per Second) and FLOPs.

```bash
# Benchmark FPS
# Syntax: python tools/analysis_tools/benchmark.py [CONFIG_FILE]
# Example: Benchmark the ALOcc-3D model
python tools/analysis_tools/benchmark.py configs/alocc/alocc_3d_256x704_bevdet_preatrain.py

# Calculate FLOPs
# Syntax: python tools/analysis_tools/get_flops.py [CONFIG_FILE] --modality image --shape 256 704
# Example: Calculate FLOPs for the ALOcc-3D model
python tools/analysis_tools/get_flops.py configs/alocc/alocc_3d_256x704_bevdet_preatrain.py --modality image --shape 256 704
```

---

## 📊 Main Results

Here are the performance benchmarks of models implemented in **OccVerse**. The results below are for the `ALOcc` series.

#### Performance on Occ3D-nuScenes (trained with camera visible mask)
| Model           | Backbone  | Input Size | mIoU<sub>D</sub><sup>m</sup> | mIoU<sup>m</sup> | FPS  |
|:----------------|:---------:|:----------:|:----------:|:----------:|:----:|
| **ALOcc-2D-mini** | ResNet-50 | 256 × 704  | 35.4       | 41.4       | 30.5 |
| **ALOcc-2D**      | ResNet-50 | 256 × 704  | 38.7       | 44.8       | 8.2  |
| **ALOcc-3D**      | ResNet-50 | 256 × 704  | 39.3       | 45.5       | 6.0  |

#### Performance on Occ3D-nuScenes (trained w/o camera visible mask)
| Method          | Backbone  | Input Size | mIoU | RayIoU | RayIoU<sub>1m, 2m, 4m</sub> | FPS  |
|:----------------|:---------:|:----------:|:----:|:------:|:-----------------:|:----:|
| **ALOcc-2D-mini** | ResNet-50 | 256 × 704  | 33.4 | 39.3   | 32.9, 40.1, 44.8  | 30.5 |
| **ALOcc-2D**      | ResNet-50 | 256 × 704  | 37.4 | 43.0   | 37.1, 43.8, 48.2  | 8.2  |
| **ALOcc-3D**      | ResNet-50 | 256 × 704  | 38.0 | 43.7   | 37.8, 44.7, 48.8  | 6.0  |

#### Performance on OpenOcc (Semantic Occupancy and Flow)
| Method            | Backbone  | Input Size | Occ Score | mAVE  | mAVE<sub>TP</sub> | RayIoU | RayIoU<sub>1m, 2m, 4m</sub> |
|:------------------|:---------:|:----------:|:---------:|:-----:|:-----------:|:------:|:-----------------:|
| **ALOcc-Flow-2D**   | ResNet-50 | 256 × 704  | 42.1      | 0.537 | 0.427       | 40.5   | 34.3, 41.3, 45.8  |
| **ALOcc-Flow-3D**   | ResNet-50 | 256 × 704  | 43.0      | 0.556 | 0.481       | 41.9   | 35.6, 42.8, 47.4  |


---

## 🤝 Contribution

We welcome contributions from the community! If you find a bug, have a feature request, or want to contribute new models to OccVerse, please feel free to open an issue or submit a pull request.

---

## 🙏 Acknowledgement

We gratefully acknowledge the foundational work of many excellent open-source projects, and we would like to extend our special thanks to:

- [open-mmlab](https://github.com/open-mmlab)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [FB-Occ](https://github.com/NVlabs/FB-BEV)
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
- [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

---

## 📜 Citation

If you find OccVerse useful in your research, please consider citing our relevant papers:

```bibtex
@InProceedings{chen2025rethinking,
    author    = {Chen, Dubing and Zheng, Huan and Fang, Jin and Dong, Xingping and Li, Xianfei and Liao, Wenlong and He, Tao and Peng, Pai and Shen, Jianbing},
    title     = {Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {1505-1515}
}

@InProceedings{chen2025alocc,
    author    = {Chen, Dubing and Fang, Jin and Han, Wencheng and Cheng, Xinjing and Yin, Junbo and Xu, Chenzhong and Khan, Fahad Shahbaz and Shen, Jianbing},
    title     = {Alocc: adaptive lifting-based 3d semantic occupancy and cost volume-based flow prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}

@InProceedings{chen2025semantic,
    author    = {Chen, Dubing and Zheng, Huan and Zhou, Yucheng and Li, Xianfei and Liao, Wenlong and He, Tao and Peng, Pai and Shen, Jianbing},
    title     = {Semantic Causality-Aware Vision-Based 3D Occupancy Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```










