import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def get_last_conv_layer_name(model):
    inner_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    target_model = inner_model if inner_model else model
    for layer in reversed(target_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_last_spatial_layer_name(model):
    # For transformers, look for the last LayerNormalization before GlobalAveragePooling
    inner_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    target_model = inner_model if inner_model else model
    for layer in reversed(target_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.LayerNormalization, tf.keras.layers.Add)):
            if len(layer.output_shape) == 4 or len(layer.output_shape) == 3:
                return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, crop_array=None):
    inner_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    
    if inner_model is not None and crop_array is not None:
        # Dual input case
        target_layer_output = inner_model.get_layer(last_conv_layer_name).output
        grad_inner_model = tf.keras.Model(inner_model.inputs, [target_layer_output, inner_model.output])
        
        image_input = model.get_layer("image_input").input
        crop_input = model.get_layer("crop_input").input
        target_out, img_features = grad_inner_model(image_input)
        
        crop_dense = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dense) and l.units == 128)
        crop_ln = next(l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization))
        concat_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate))
        merged_dense = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dense) and l.units == 256)
        dropout_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dropout))
        output_dense = model.get_layer("disease_output")
        
        x_crop = crop_dense(crop_input)
        x_crop = crop_ln(x_crop)
        merged = concat_layer([img_features, x_crop])
        x = merged_dense(merged)
        x = dropout_layer(x)
        preds = output_dense(x)
        
        grad_model = tf.keras.Model(inputs=[image_input, crop_input], outputs=[target_out, preds])
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model([img_array, crop_array])
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    else:
        # Standard case
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
            pass 

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Create a superimposed image of heatmap over original image."""
    img = np.array(img_array) * 255.0
    img = img.astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * heatmap)
    
    if len(heatmap_uint8.shape) == 2:
        heatmap_uint8 = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap_colored * alpha + img
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed

def generate_explainability(model, test_gen, model_name, seed, save_dir="results", images_per_class=5):
    """
    Generate GradCAM/Attention overlays for multiple images per class.
    Picks up to `images_per_class` images from each class in the test generator.
    Saves overlays organized by class under save_dir/model_name/.
    """
    model_dir = os.path.join(save_dir, model_name)
    explainability_dir = os.path.join(model_dir, f"seed{seed}_explainability")
    os.makedirs(explainability_dir, exist_ok=True)
    
    # Determine the right layer
    if "ResNet" in model_name or "EfficientNet" in model_name or "MobileNet" in model_name:
        layer_name = get_last_conv_layer_name(model)
        prefix = "GradCAM"
    else:
        layer_name = get_last_spatial_layer_name(model)
        prefix = "AttnMap"
        
    if layer_name is None:
        print(f"Could not find a suitable layer for explainability in {model_name}.")
        return

    # Get class names from the generator
    class_indices = test_gen.class_indices  # {'Corn/Corn___Common_Rust': 0, ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)
    
    # Collect images per class
    class_images = {i: [] for i in range(num_classes)}
    
    # Iterate through the test generator to collect images
    steps = len(test_gen)
    for step_idx in range(steps):
        batch_data, batch_labels = test_gen[step_idx]
        batch_class_indices = np.argmax(batch_labels, axis=1)
        
        is_dual_input = isinstance(batch_data, tuple)
        if is_dual_input:
            batch_imgs = batch_data[0]
            batch_crops = batch_data[1]
        else:
            batch_imgs = batch_data
            batch_crops = None
            
        for i in range(len(batch_imgs)):
            cls_idx = batch_class_indices[i]
            if len(class_images[cls_idx]) < images_per_class:
                if is_dual_input:
                    class_images[cls_idx].append((batch_imgs[i], batch_crops[i]))
                else:
                    class_images[cls_idx].append(batch_imgs[i])
        
        # Check if all classes have enough images
        if all(len(v) >= images_per_class for v in class_images.values()):
            break
    
    # Generate overlays for each class
    for cls_idx, images in class_images.items():
        if not images:
            continue
            
        class_name = idx_to_class.get(cls_idx, f"class_{cls_idx}")
        # Clean up class name for folder (replace / with _)
        clean_class_name = class_name.replace("/", "_").replace(" ", "_")
        class_dir = os.path.join(explainability_dir, clean_class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for item_idx, item in enumerate(images):
            try:
                if is_dual_input:
                    img, crop = item
                    img_batch = np.expand_dims(img, axis=0)
                    crop_batch = np.expand_dims(crop, axis=0)
                else:
                    img = item
                    img_batch = np.expand_dims(img, axis=0)
                    crop_batch = None
                    
                heatmap = make_gradcam_heatmap(img_batch, model, layer_name, crop_array=crop_batch)
                overlay = overlay_heatmap(img, heatmap)
                
                # Save overlay
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
                axes[0].set_title("Original")
                axes[0].axis("off")
                
                # Heatmap
                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title(f"{prefix} Heatmap")
                axes[1].axis("off")
                
                # Overlay
                axes[2].imshow(overlay)
                axes[2].set_title(f"{prefix} Overlay")
                axes[2].axis("off")
                
                plt.suptitle(f"{model_name} | {class_name} | Image {item_idx+1}", fontsize=12)
                plt.tight_layout()
                save_path = os.path.join(class_dir, f"{prefix}_img{item_idx+1}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
                
            except Exception as e:
                print(f"  Failed for {class_name} image {item_idx+1}: {e}")
    
    print(f"Explainability maps saved to: {explainability_dir}")
