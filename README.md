# Multi-Modal Carpal Tunnel Syndrome Segmentation (MCTS-Net)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Expert-level MRI Segmentation for Carpal Tunnel Syndrome (CTS) Diagnosis.** > A Deep Learning framework utilizing Multi-Modal U-Net to precisely isolate the Median Nerve, Flexor Tendons, and Carpal Tunnel from T1/T2 MRI sequences.

---

## 📖 Overview

**Carpal Tunnel Syndrome (CTS)** is a common neuropathy caused by the compression of the median nerve within the carpal tunnel. Accurate segmentation of wrist tissues from Magnetic Resonance Imaging (MRI) is a critical step for automated clinical diagnosis.

This project implements a **Multi-Modal U-Net** architecture that fuses **T1-weighted** and **T2-weighted** MRI images to segment three key anatomical structures:

1.  **Median Nerve (MN)** 🟡
2.  **Flexor Tendons (FT)** 🔵
3.  **Carpal Tunnel (CT)** 🔴

The system includes a **professional-grade GUI** for interactive inference, allowing researchers to visualize segmentation results with real-time Dice Score calculation and slice navigation.

## Key Features

* [cite_start]**Multi-Modal Fusion**: Takes both **T1** and **T2** MRI images as a 2-channel input to maximize tissue contrast and segmentation accuracy[cite: 37, 38].
* **Robust U-Net Architecture**: Features Bilinear Upsampling and Double Convolution blocks for high-resolution boundary detection.
* **Efficient Training**: Implements **Mixed Precision Training (AMP)** and **5-Fold Cross Validation** for optimal performance on consumer-grade GPUs (e.g., RTX 4060).
* **Interactive Player (GUI)**:
    * Slider-based MRI slice navigation.
    * Real-time Dice Coefficient (DSC) calculation.
    * Color-coded segmentation overlays (Ground Truth vs. AI Prediction).

## Project Structure

The project code is organized in the `src/` directory to maintain a clean root environment:

```text
.
├── src/
│   ├── train.py            # Main training script (U-Net with 5-Fold CV)
│   ├── inference_gui.py    # Interactive GUI for visualization
│   └── dataset.zip         # (Place your dataset here, script auto-unzips it)
├── checkpoints/            # Stores trained models (*.pth)
├── requirements.txt        # Python dependencies
└── README.md
