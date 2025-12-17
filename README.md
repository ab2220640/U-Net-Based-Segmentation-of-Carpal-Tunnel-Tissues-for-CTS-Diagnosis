# U-Net-Based-Segmentation-of-Carpal-Tunnel-Tissues-for-CTS-Diagnosis
Expert-level MRI Segmentation for Carpal Tunnel Syndrome (CTS). Uses U-Net to isolate Median Nerve &amp; Flexor Tendons with >90% accuracy (DSC). Includes an interactive slider-based GUI for visualization.

# MRI Carpal Tunnel Segmentation

This project implements a Deep Learning solution to segment wrist tissues (Median Nerve, Flexor Tendons, Carpal Tunnel) from MRI images.

##  Project Structure
* **`train.py`**: The main training script (U-Net architecture).
* **`inference_gui.py`**: The Graphical User Interface (GUI) for visualization.
* **`dataset.zip`**: Compressed MRI dataset (Contains T1/T2 images and Ground Truth).

##  How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
###2. Prepare the Dataset
Important: Please unzip the dataset before running the code.
```bash
unzip dataset.zip
# Ensure the extracted folder is named 'dataset' or matches the path in train.py
