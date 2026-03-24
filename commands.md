# End-to-End Execution Commands

This file contains the sequential instructions to set up, load data, and run the machine learning pipelines on your Mac. You can copy-paste these commands directly into your terminal.

## 1. Structure the Dataset Directory

Make sure your images are located inside `data/` and structure them as standard categorical folders for training and validation datasets.

```bash
# General Directory Structure Required
#
# CvT-crop-disease-comparison/
# └── data/
#     ├── training/
#     │   ├── blight/
#     │   ├── rust/
#     │   └── healthy/
#     └── validation/
#         ├── blight/
#         ├── rust/
#         └── healthy/
```

## 2. Enter the Working Directory Environment

We've already established a Python Virtual Environment (`env`) with GPU compatibility via `tensorflow-metal` and `tensorflow-macos` specifically built for Apple Silicon. Load it using:

```bash
source env/bin/activate
```

## 3. Training & Running Benchmarks

The `train.py` orchestrator supports compiling all experiments (across 3 uniform random seeds each), tracking timing and precision metrics instantly, and saving them to independent files to stop overriding runs explicitly.

**Option A: Run everything (Warning: Can take a while depending on Data)**

```bash
python train.py --data_dir data --epochs 10 --batch_size 16
```

**Option B: Run an Individual Model separately**

Since there is a `--model` flag, you can orchestrate tasks cleanly in parallel on different terminals, or sequentially. The script prevents output collisions by assigning the name to output binaries `(e.g., benchmark_results_ResNet50.json)`:

```bash
# Examples:
python train.py --model ResNet50 --data_dir data --epochs 10 --batch_size 16
python train.py --model CvT --data_dir data --epochs 10 --batch_size 16
python train.py --model SwinTiny --data_dir data --epochs 10 --batch_size 16
```

**Option C: Run a Specific Seed**

You can also run a specific seed (e.g., 42, 123, or 456) instead of running all three seeds by using the `--seed` flag:

```bash
python train.py --model CvT --seed 42 --data_dir data --epochs 10 --batch_size 16
```

## 4. Verification Check 

If you simply want to make sure the loop runs on the hardware compilation securely before plugging in large crop files, you can use the dummy test flag. It synthesizes blank static noise and passes it to check gradients:

```bash
python train.py --model CvT --dummy --epochs 1 --batch_size 2
```

## 5. Checking Outputs

All diagrams explicitly show what was derived from which model. You will find:

- Visual Loss & Accuracy: `results/[modelName]_seed[seed]_accuracy.png` and `_loss.png`
- Detailed Heatmaps: `results/[modelName]_seed[seed]_GradCAM.png` or `_AttnMap.png`
- Core JSON Stat file: `results/benchmark_results_[modelName]_seed[seed].json`

