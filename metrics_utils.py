import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from matplotlib.ticker import MaxNLocator

def get_flops(model):
    """Calculate FLOPs for a given tf.keras model, supporting multi-input."""
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        
        # Determine number of inputs and their shapes
        concrete_inputs = []
        for inp in model.inputs:
            concrete_inputs.append(tf.TensorSpec([1] + list(inp.shape[1:]), inp.dtype))
            
        real_model = tf.function(model).get_concrete_function(concrete_inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                              run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs for {model.name}. {e}")
        return 0

def plot_history(history, model_name, seed, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} (Seed {seed}) - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_seed{seed}_accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} (Seed {seed}) - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
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
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": float(acc), 
        "precision": float(prec), 
        "recall": float(rec), 
        "f1_score": float(f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm
    }
