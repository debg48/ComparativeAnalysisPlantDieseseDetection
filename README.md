# Crop Disease Model Comparison Project

This repository provides an automated benchmarking and comparison pipeline for Deep Learning image classification models, specializing in agricultural crop disease datasets. It natively supports Apple Silicon (M-series Macs) using the TensorFlow Metal plugin.

## Supported Models

- **ResNet50** (CNN)
- **EfficientNetB0** (CNN)
- **MobileNetV2** (CNN)
- **ViT** (Vision Transformer, Custom Small Definition)
- **SwinTiny** (Hierarchical Transformer, Customized)
- **CvT** (Convolutional Vision Transformer, hybrid architecture)

## Analysis & Output Generated

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Computational Profile**: Train/Inference duration, Network Parameter count, and detailed FLOP operations.
- **Robustness Sandbox**: Out-of-the-box evaluations on corrupted datasets (Gaussian Noise, Gaussian Blur, Lighting/Brightness).
- **Explainability**: Generates Grad-CAM visual heatmaps specifically tracing back focal points influencing decisions inside the classification architectures. Supported for both CNNs and Transformers.
- **Hierarchical Classification (Two-Stage)**: A novel approach that first identifies the crop type (Phase 1: Crop Router) and then uses a single, dual-input specialist (Phase 2) conditioned on the crop label to identify the disease. This architecture enables feature sharing across different crops while maintaining hierarchical precision.
- **Comparative Analysis**: Tools to automatically compare the performance, efficiency, and robustness of the Global vs. Hierarchical approaches.
- **Experiments Report**: Generates a unified Markdown file displaying Baseline Comparison, Phase-wise Evaluation, Error Propagation, Per-Class details, and Computational Analysis tradeoffs.
- **Robustness Sandbox**: Evaluations on corrupted datasets (Gaussian Noise, Blur, Brightness) to test model stability.

## Structure

- `models.py`: Network construction scripts for all 6 architectures, including the Dual-Input Phase 2 model.
- `data_loader.py`: TF Preprocessing pipelines and Generators for the global baseline.
- `hierarchical_data_loader.py`: Specialized data loader for the two-stage system, featuring the `DualInputWrapper` for joint image/label training.
- `metrics_utils.py`: Analytical engines evaluating runtime, matrices, and plotting visuals.
- `explainability.py`: Grad-CAM extraction methods with per-class overlay generation.
- `train.py`: Primary loop runner for the Global (Flat) baseline.
- `train_hierarchical.py`: Orchestrator for the Two-Stage Hierarchical system (Router + Joint Specialist).
- `compare_results.py`: Comparison engine that generates global-vs-hierarchical charts and error propagation analysis.
- `generate_experiments_report.py`: Aggregates all global and hierarchical JSON benchmarks into a definitive 5-Group Markdown report.

## Getting Started

See the inner `commands.md` file for an active walkthrough to load your dataset in `data/`, setup requirements natively, and initiate evaluations.
