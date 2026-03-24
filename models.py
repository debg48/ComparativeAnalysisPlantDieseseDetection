import tensorflow as tf
from tensorflow.keras import layers as L

def conv_embedding(x, filters, kernel_size, strides):
    x = L.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Reshape((-1, filters))(x)
    return x

def patch_embedding(x, patch_size, projection_dim):
    x = L.Conv2D(projection_dim, patch_size, strides=patch_size, padding="valid")(x)
    x = L.Reshape((-1, projection_dim))(x)
    return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation=tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, projection_dim, num_heads, name=None):
    x1 = L.LayerNormalization(epsilon=1e-6)(x)
    attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, name=name)(x1, x1)
    x2 = L.Add()([attn, x])
    x3 = L.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, [projection_dim*2, projection_dim], 0.1)
    return L.Add()([x3, x2])

def get_resnet50(input_shape, num_classes):
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    outputs = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, outputs, name="ResNet50")

def get_efficientnet_b0(input_shape, num_classes):
    base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    outputs = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, outputs, name="EfficientNetB0")

def get_mobilenet_v2(input_shape, num_classes):
    base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    outputs = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, outputs, name="MobileNetV2")

def get_vit_small(input_shape, num_classes):
    inputs = L.Input(shape=input_shape)
    # Simple ViT small approximation
    patch_size = 16
    projection_dim = 128
    num_heads = 4
    transformer_layers = 4
    
    x = patch_embedding(inputs, patch_size, projection_dim)
    
    # Adding positional embeddings
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    pos_embed = L.Embedding(input_dim=num_patches, output_dim=projection_dim)(tf.range(start=0, limit=num_patches, delta=1))
    x = x + pos_embed
    
    for i in range(transformer_layers):
        x = transformer_block(x, projection_dim, num_heads, name=f"vit_attn_{i}")
        
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ViT_Small")

def get_swin_tiny(input_shape, num_classes):
    # Simplified hierarchical transformer mimicking Swin Tiny architecture
    inputs = L.Input(shape=input_shape)
    
    # Stage 1
    x = patch_embedding(inputs, patch_size=4, projection_dim=48)
    x = transformer_block(x, projection_dim=48, num_heads=3, name="swin_stage1_attn")
    
    # Mimic patch merging by reshaping and downsampling
    # For a real Swin, this is complex shifted windows, but we use standard MHSA for brevity
    # Stage 2
    x = L.Dense(96)(x)
    x = transformer_block(x, projection_dim=96, num_heads=3, name="swin_stage2_attn")
    
    # Stage 3
    x = L.Dense(192)(x)
    x = transformer_block(x, projection_dim=192, num_heads=6, name="swin_stage3_attn")
    x = transformer_block(x, projection_dim=192, num_heads=6, name="swin_stage3_attn_2")
    
    # Stage 4
    x = L.Dense(384)(x)
    x = transformer_block(x, projection_dim=384, num_heads=12, name="swin_stage4_attn")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="SwinTiny_Simplified")

def get_cvt_model(input_shape, num_classes):
    inputs = L.Input(shape=input_shape)
    x = conv_embedding(inputs, filters=64, kernel_size=7, strides=4)
    x = transformer_block(x, projection_dim=64, num_heads=4, name="cvt_attn_0")
    x = transformer_block(x, projection_dim=64, num_heads=4, name="cvt_attn_1")
    x = transformer_block(x, projection_dim=64, num_heads=4, name="cvt_attn_2")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="CvT_Custom")

def get_model(model_name, input_shape, num_classes):
    if model_name == "ResNet50":
        return get_resnet50(input_shape, num_classes)
    elif model_name == "EfficientNetB0":
        return get_efficientnet_b0(input_shape, num_classes)
    elif model_name == "MobileNetV2":
        return get_mobilenet_v2(input_shape, num_classes)
    elif model_name == "ViT":
        return get_vit_small(input_shape, num_classes)
    elif model_name == "SwinTiny":
        return get_swin_tiny(input_shape, num_classes)
    elif model_name == "CvT":
        return get_cvt_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
