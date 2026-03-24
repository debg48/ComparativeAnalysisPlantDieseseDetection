import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_last_spatial_layer_name(model):
    # For transformers, look for the last LayerNormalization before GlobalAveragePooling
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.LayerNormalization, tf.keras.layers.Add)):
            if len(layer.output_shape) == 4 or len(layer.output_shape) == 3:
                return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool gradients over the spatial dimensions depending on shape
    if len(grads.shape) == 4: # CNNs (batch, h, w, c)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    else: # Transformers (batch, patches, c)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    last_conv_layer_output = last_conv_layer_output[0]
    
    # Weight the output feature map
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For transformers, reshape patches to 2D grid
    if len(heatmap.shape) == 1:
        size = int(np.sqrt(heatmap.shape[0]))
        if size * size == heatmap.shape[0]:
            heatmap = tf.reshape(heatmap, (size, size))
        else:
            # Fallback if it can't be perfectly squared (e.g. includes CLS token)
            # Remove CLS if present (just an example, our ViT doesn't use CLS token)
            pass 

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_array, heatmap, save_path, alpha=0.4):
    img = np.array(img_array[0]) * 255.0
    img = img.astype(np.uint8)

    heatmap = np.uint8(255 * heatmap)
    
    if len(heatmap.shape) == 2:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_explainability(model, img_array, model_name, seed, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    if "ResNet" in model_name or "EfficientNet" in model_name or "MobileNet" in model_name:
        layer_name = get_last_conv_layer_name(model)
        prefix = "GradCAM"
    else:
        layer_name = get_last_spatial_layer_name(model)
        prefix = "AttnMap"
        
    if layer_name is None:
        print(f"Could not find a suitable layer for explainability in {model_name}.")
        return

    heatmap = make_gradcam_heatmap(img_array, model, layer_name)
    save_path = os.path.join(save_dir, f"{model_name}_seed{seed}_{prefix}.png")
    save_and_display_gradcam(img_array, heatmap, save_path)
