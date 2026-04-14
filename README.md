# Crop Disease Model Comparison Project

This repository provides an automated benchmarking and comparison pipeline for Deep Learning image classification models, specializing in agricultural crop disease datasets. It natively supports Apple Silicon (M-series Macs) using the TensorFlow Metal plugin.

## Supported Models

- **ResNet50** / **EfficientNetB0** / **MobileNetV2** (CNN Baselines)
- **ViT** / **SwinTiny** (Pure Transformers)
- **CvT** (Convolutional Vision Transformer - Hybrid)
- **SuperConformer** (State-of-the-Art Hybrid with Multi-Scale Fusion and Dynamic Cross-Attention)

## Core Technologies

- **Hierarchical Pipeline**: A two-stage system (Phase 1: Router, Phase 2: Joint Specialist) that achieves >90% accuracy by conditioning disease detection on crop identity.
- **Dynamic Cross-Attention**: Replaces static token concatenation with a mathematically pure query-key-value attention mechanism for crop conditioning.
- **Multi-Scale Feature Fusion**: Simultaneous extraction of fine 8x8 and coarse 16x16 tokens to capture both microscopic fungal patterns and macroscopic leaf structure.
- **Explainability**: Custom Grad-CAM implementation supporting dual-input transformer architectures.
- **Apple Silicon Optimized**: Native acceleration via the TensorFlow Metal plugin.

## Recent Benchmarking Results (E2E)

| Architecture | Accuracy | F1-Score | Corn (Specialist) | Potato (Specialist) |
| :--- | :--- | :--- | :--- | :--- |
| **CvT Hierarchical** | 89.94% | 0.8981 | 92.20% | 98.76% |
| **SuperConformer** | **90.04%** | **0.9001** | **95.15%** | **99.38%** |

*Note: The SuperConformer architecture specifically solved the fine-grained ambiguity in Corn, pushing it into the 95% elite bracket.*

## Usage

### Phase 1: Global Baseline

```bash
python3 train.py --model CvT --batch_size 16
```

### Phase 2: Hierarchical (Router + Specialist)

```bash
python3 train_hierarchical.py --model Conformer --batch_size 8
```

## Structure

- `models.py`: Network construction for all architectures, including the **SuperConformer** and **Dual-Input** logic.
- `train.py`: Standard single-phase training pipeline with data augmentation, label smoothing, and callbacks.
- `train_hierarchical.py`: Orchestrator for the Two-Stage system (Router + Joint Specialist) with prediction masking.
- `data_loader.py` & `hierarchical_data_loader.py`: Dataset loading and preprocessing pipelines.
- `explainability.py`: Grad-CAM extraction with native support for dual-input conditioning.
- `metrics_utils.py`: Analytical engines evaluating runtime, matrices, and FLOPs.
- `compare_results.py`: Tools for comparative evaluation of performance metrics across all architectures.
- `generate_experiments_report.py`: Automated generation of detailed markdown experiment reports.
- `plot_single_phase_accuracy.py` & `plot_model_parameters.py`: Visualization scripts for benchmarking and analysis.

