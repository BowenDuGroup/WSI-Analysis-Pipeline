# WSI Analysis Pipeline: DINO + CLAM

This repository implements a complete deep learning pipeline for Whole Slide Image (WSI) analysis in computational pathology. It integrates **DINO (Self-Supervised Learning)** for label-free feature learning and **CLAM (Clustering-constrained Attention Multiple instance learning)** for weakly supervised slide-level classification.

## 📋 Pipeline Overview

The pipeline consists of three main phases:
1.  **Pre-training (DINO):** Train a Vision Transformer (ViT-Small) on pathology patches using self-supervised learning.
2.  **Feature Extraction:** Use the pre-trained DINO Teacher backbone to extract 384-dimensional feature vectors from WSI patches.
3.  **Classification (CLAM):** Train the CLAM-SB model using the extracted features for slide-level prediction (e.g., High-risk tumor vs. Low-risk tumor).

## 📂 Project Structure

```text
.
├── dino_train/                   # Phase 1: DINO Self-Supervised Training
│   ├── main_dino.py              # Main training script
│   ├── model_dino.py             # ViT backbone definitions
│   ├── data_dino.py              # Data augmentation pipeline
│   ├── loss_dino.py              # DINO loss implementation
│   └── wsi_to_patch.py           # WSI process
├── clam_train/                   # Phase 3: CLAM Weakly Supervised Training
│   ├── train.py                  # Main training script
│   ├── model_clam.py             # CLAM-SB model architecture
│   └── data_clam.py              # Dataset and training loops
├── utils/                        # Shared utilities
│   └── extract_features.py       # Phase 2: Feature extraction script
├── data/                         # Data storage
│   └── dataset.csv               # Slide labels
├── models/                       # Dino model
│   ├── model_dino_teacher.pth    # Clam model
│   └── model_clam.pth            # Slide labels
├── requirements.txt              # Python dependencies
├── environment.yml               # Python dependencies
└── README.md

```

## 📋 Workflow

![Pipeline Architecture](./images/workflow.jpg)