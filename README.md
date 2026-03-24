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
- **Explainability**: Generates Grad-CAM visual heatmaps specifically tracing back focal points influencing decisions inside the classification architectures.

## Structure
- `models.py`: Network construction scripts for the models above.
- `data_loader.py`: TF Preprocessing pipelines and Generators including Custom robustness modifier functions.
- `metrics_utils.py`: Analytical engines evaluating runtime, matrices, and plotting visuals.
- `explainability.py`: Grad-CAM extraction methods for both simple CNNs and Transformers' `LayerNormalization` boundaries.
- `train.py`: Primary loop runner script.

## Getting Started
See the inner `commands.md` file for an active walkthrough to load your dataset in `data/`, setup requirements natively, and initiate evaluations.
