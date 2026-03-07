

# 📦 Parcel Damage Detection for Conveyor Systems

This repository contains an end-to-end computer vision pipeline designed to detect **damaged vs. normal shipping packages** in real-time on industrial conveyor belts. The system utilizes a **YOLOv8n (nano)** model architecture, optimized for high-speed edge inference.

## 🛠️ System Architecture

* **Model:** YOLOv8n (2.10.0+cu128).
* **Dataset:** Multi-source dataset (Kaggle, Roboflow, Hugging Face) capped at 10,000 images.
* **Target Hardware:** NVIDIA Jetson Orin (TensorRT FP16) and MacBook M2 (Webcam testing).

---

## 🚩 Current Issues

Based on the current pipeline implementation, the following issues have been identified:

1. **Heuristic-Based Labeling Risks:** The system uses a `heuristic_class` method to recover labels from original folder-level keywords (e.g., 'damaged', 'broken'). If the source data is mislabeled or lacks a clear directory structure, this can lead to high label noise.
2. **Dataset Imbalance:** While the pipeline samples up to 10,000 images, the actual download from Kaggle yielded 12,954 images, meaning significant data is discarded without a clear strategy for class balancing.
3. **Dependency on API Keys:** The notebook relies heavily on external platform secrets (`KAGGLE_KEY`, `ROBOFLOW_API_KEY`). If these APIs are throttled or keys expire, the training pipeline fails immediately.
4. **Hardware Bottlenecks:** Although optimized for A100 GPUs during training, the target deployment is the NVIDIA Jetson Orin. Differences in CUDA versions or memory constraints on edge devices can lead to performance degradation if not validated during the conversion process.

---

## 🚀 Suggested Improvements

To move this system from a prototype to a production-ready solution, the following enhancements are recommended:

### 1. Advanced Data Augmentation

Integrate **Albumentations** more deeply into the pipeline to simulate industrial environments. Specifically, add:

* **Motion Blur:** To simulate fast-moving conveyor belts.
* **Specular Reflection:** To mimic plastic-wrapped parcels under warehouse lighting.

[Image showing motion blur and lighting augmentation on shipping boxes]

### 2. Robust Validation Strategy

* **Stratified Sampling:** Replace random sampling with stratified sampling to ensure the `damaged_package` class is represented proportionally in both training and validation sets.
* **Cross-Validation:** Implement k-fold cross-validation to ensure model stability across different dataset sources (Roboflow vs. Kaggle).

### 3. Deployment Optimization

* **Int8 Quantization:** Move from FP16 to **Int8 quantization** for the TensorRT export. This would significantly reduce latency on the Jetson Orin with minimal accuracy loss.
* **Automated Export Validation:** Add a post-training script to run `best_model.onnx` through a latency benchmark on the target hardware before final deployment.

### 4. Hybrid Labeling Verification

* Implement a **Semi-Automated Labeling** workflow where the model’s low-confidence predictions are flagged for manual review, reducing reliance on the current folder-based heuristic.
