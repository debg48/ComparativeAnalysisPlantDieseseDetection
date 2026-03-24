import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def get_flops(model):
    """Calculate FLOPs for a given tf.keras model."""
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        real_model = tf.function(model).get_concrete_function(tf.TensorSpec([1] + model.inputs[0].shape[1:], model.inputs[0].dtype))
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        # We need to manually count FLOPs using the profiler
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                              run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs. {e}")
        return 0

def plot_history(history, model_name, seed, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} (Seed {seed}) - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_seed{seed}_accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} (Seed {seed}) - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_seed{seed}_loss.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name, seed, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} (Seed {seed}) - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f"{model_name}_seed{seed}_confusion_matrix.png"))
    plt.close()

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
