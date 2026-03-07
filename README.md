

# 📦 Parcel Damage Detection for Conveyor Systems

This project implements an end-to-end computer vision pipeline to detect **damaged vs. normal shipping packages** in real-time. It is designed for industrial environments, optimized for high-speed edge inference on the NVIDIA Jetson Orin.

**🚀 Live Demo:** [Hugging Face Space - PackageCam](https://huggingface.co/spaces/chernhaw1/packagecam)

---

## 🏗️ Model Building Pipeline

The pipeline is automated within a single Jupyter environment, moving from raw multi-source data to optimized deployment weights.

1. **Environment Setup**: Configuration of CUDA 12.8, PyTorch 2.10.0, and Ultralytics YOLOv8.
2. **Dataset Collection**:
* Aggregates up to 10,000 images from **Kaggle**, **Hugging Face**, and **Roboflow**.
* Uses a `heuristic_class` method to recover labels from directory keywords like 'damaged' or 'broken'.


3. **Preprocessing & Augmentation**:
* Implements **Albumentations** for robust training.
* Automatically generates YOLO-format labels for folder-labeled images.


4. **Training**:
* Architecture: **YOLOv8n (Nano)**.
* Hardware: Trained on **NVIDIA A100-SXM4** (40GB VRAM) for rapid convergence.


5. **Export & Deployment**:
* Converts PyTorch (`.pt`) weights to **ONNX** format for Jetson TensorRT compatibility.
* Includes scripts for MacBook M2 webcam testing and Jetson Orin production inference.



---

## 🚩 Current Issues

Based on the latest training run, the following technical issues were identified:

* **Label Heuristic Risks**: The current system relies on original folder paths (via `.src` sidecar files) to determine the class. If the source dataset has inconsistent folder naming, it can introduce significant label noise.
* **Dataset Sampling**: The pipeline caps data at 10,000 images but found 12,954 raw images in Kaggle alone. This random sampling may exclude critical "edge case" examples of damage.
* **Dependency Fragility**: The pipeline depends on four separate external APIs (Kaggle, HF, Roboflow, Weights & Biases). A failure or credential expiration in any one of these can break the automated training loop.

---

## 🚀 Suggested Improvements

To enhance the model's reliability for production, the following improvements are recommended:

* **Int8 Quantization**: Move from FP16 to **Int8 quantization** during the TensorRT export. This would double the inference speed on the Jetson Orin with negligible accuracy loss.
* **Synthetic Data Generation**: Since real "damaged" package data can be rare, use Generative AI or 3D rendering to create synthetic "crushed" or "torn" box images to better balance the classes.
* **Automated Validation Plots**: Ensure the `evaluation_plots.png` (Confusion Matrix and PR Curve) are reviewed after every run to identify if the model is confusing specific types of damage (e.g., tape vs. a tear).
* **Hardware-in-the-Loop Testing**: Integrate the `jetson_inference.py` script into a CI/CD pipeline to verify performance directly on the target Orin hardware before deployment.
### 4. Hybrid Labeling Verification

* Implement a **Semi-Automated Labeling** workflow where the model’s low-confidence predictions are flagged for manual review, reducing reliance on the current folder-based heuristic.
