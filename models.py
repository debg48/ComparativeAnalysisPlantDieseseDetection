import tensorflow as tf
from tensorflow.keras import layers as L

def get_cnn_router(input_shape, num_classes):
    """CNN for Phase 1 crop routing. 5 conv blocks for high-accuracy 5-class routing."""
    inputs = L.Input(shape=input_shape)
    
    x = L.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.MaxPooling2D(2)(x)
    
    x = L.Conv2D(64, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.MaxPooling2D(2)(x)
    
    x = L.Conv2D(128, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.MaxPooling2D(2)(x)
    
    x = L.Conv2D(256, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.MaxPooling2D(2)(x)
    
    x = L.Conv2D(512, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.GlobalAveragePooling2D()(x)
    
    x = L.Dropout(0.3)(x)
    x = L.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, x, name="CNN_Router")

def conv_embedding(inputs, filters, kernel_size, strides, name="conv_embed"):
    x = L.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", name=name)(inputs)
    x = L.LayerNormalization(epsilon=1e-6)(x)
    batch = tf.shape(x)[0]
    seq_len = x.shape[1] * x.shape[2]
    # Reshape specifically for TF compatibility dynamically
    x = tf.reshape(x, [batch, seq_len, filters])
    return x

def robust_conv_embedding(inputs, filters, name="robust_conv_embed"):
    # Stage 1: stride 4 (224 -> 56) with overlapping kernel
    x = L.Conv2D(filters=filters//2, kernel_size=7, strides=4, padding="same", activation="gelu", name=f"{name}_s1")(inputs)
    x = L.LayerNormalization(epsilon=1e-6)(x)
    # Stage 2: stride 4 (56 -> 14) to hit the 14x14 grid, preserving dense features
    x = L.Conv2D(filters=filters, kernel_size=4, strides=4, padding="same", activation="gelu", name=f"{name}_s2")(x)
    x = L.LayerNormalization(epsilon=1e-6)(x)
    
    batch = tf.shape(x)[0]
    seq_len = x.shape[1] * x.shape[2]
    x = tf.reshape(x, [batch, seq_len, filters])
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

def classic_transformer_block(x, projection_dim, num_heads, name=None):
    x1 = L.LayerNormalization(epsilon=1e-6)(x)
    # The fix: key_dim is per head. To match standard Transformers and the RoPE implementation, 
    # key_dim = projection_dim // num_heads.
    key_dim = projection_dim // num_heads
    attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=name)(x1, x1)
    x2 = L.Add()([attn, x])
    x3 = L.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, [projection_dim*2, projection_dim], 0.1)
    return L.Add()([x3, x2])

class RoPE2DAttention(tf.keras.layers.Layer):
    """Multi-Head Attention with 2D Rotary Position Embeddings.
    Splits each head's dimension in half: one half encodes row position,
    the other encodes column position via sinusoidal rotations on Q and K.
    """
    def __init__(self, projection_dim, num_heads, grid_height, grid_width, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.head_dim = projection_dim // num_heads
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        self.q_dense = L.Dense(projection_dim)
        self.k_dense = L.Dense(projection_dim)
        self.v_dense = L.Dense(projection_dim)
        self.out_dense = L.Dense(projection_dim)

    def build(self, input_shape):
        super().build(input_shape)
        half_head = self.head_dim // 2
        quarter = half_head // 2

        freq = 1.0 / (10000.0 ** (tf.cast(tf.range(quarter), tf.float32) / tf.cast(quarter, tf.float32)))

        rows = tf.cast(tf.repeat(tf.range(self.grid_height), self.grid_width), tf.float32)
        cols = tf.cast(tf.tile(tf.range(self.grid_width), [self.grid_height]), tf.float32)

        angles_row = tf.einsum('i,j->ij', rows, freq)  # (seq, quarter)
        angles_col = tf.einsum('i,j->ij', cols, freq)

        self.cos_row = tf.cos(angles_row)
        self.sin_row = tf.sin(angles_row)
        self.cos_col = tf.cos(angles_col)
        self.sin_col = tf.sin(angles_col)

    def _rotate(self, x, cos, sin):
        """Apply rotary embedding to paired elements."""
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        r_even = x_even * cos - x_odd * sin
        r_odd = x_even * sin + x_odd * cos
        stacked = tf.stack([r_even, r_odd], axis=-1)
        return tf.reshape(stacked, tf.shape(x))

    def _apply_rope_2d(self, x):
        """x: (batch, heads, seq, head_dim)"""
        half = self.head_dim // 2
        x_row = x[..., :half]
        x_col = x[..., half:]
        x_row = self._rotate(x_row, self.cos_row, self.sin_row)
        x_col = self._rotate(x_col, self.cos_col, self.sin_col)
        return tf.concat([x_row, x_col], axis=-1)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq = tf.shape(x)[1]

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        # (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = tf.transpose(tf.reshape(q, [batch, seq, self.num_heads, self.head_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [batch, seq, self.num_heads, self.head_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [batch, seq, self.num_heads, self.head_dim]), [0, 2, 1, 3])

        q = self._apply_rope_2d(q)
        k = self._apply_rope_2d(k)

        attn = tf.matmul(q, k, transpose_b=True) / self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, seq, self.projection_dim])
        return self.out_dense(out)

class ConvPositionEncoding(tf.keras.layers.Layer):
    """Convolutional Position Encoding (CPE).
    Reshapes 1D token sequence back to 2D grid, applies depthwise convolution
    to inject local spatial structure, then reshapes back to 1D.
    Critical for fine-grained lesion discrimination (e.g., Rice diseases).
    """
    def __init__(self, dim, grid_height, grid_width, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.dwconv = L.DepthwiseConv2D(
            kernel_size=kernel_size, padding='same', use_bias=True
        )

    def call(self, x):
        batch = tf.shape(x)[0]
        # (batch, seq, dim) -> (batch, H, W, dim)
        x_2d = tf.reshape(x, [batch, self.grid_height, self.grid_width, self.dim])
        x_2d = self.dwconv(x_2d)
        # (batch, H, W, dim) -> (batch, seq, dim)
        return tf.reshape(x_2d, [batch, self.grid_height * self.grid_width, self.dim])

def rope_transformer_block(x, projection_dim, num_heads, grid_h, grid_w, name=None, cpe_kernel=3):
    """Transformer block with 2D RoPE attention + Convolutional Position Encoding."""
    x1 = L.LayerNormalization(epsilon=1e-6)(x)
    attn = RoPE2DAttention(projection_dim, num_heads, grid_h, grid_w, name=name)(x1)
    x2 = L.Add()([attn, x])
    
    # CPE: Inject local spatial structure after self-attention
    x2 = x2 + ConvPositionEncoding(projection_dim, grid_h, grid_w, kernel_size=cpe_kernel, name=f"{name}_cpe" if name else None)(x2)
    
    x3 = L.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, [projection_dim * 2, projection_dim], 0.1)
    return L.Add()([x3, x2])

def get_resnet50(input_shape, num_classes, include_top=True):
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    if include_top:
        x = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, x, name="ResNet50")

def get_efficientnet_b0(input_shape, num_classes, include_top=True):
    base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    if include_top:
        x = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, x, name="EfficientNetB0")

def get_mobilenet_v2(input_shape, num_classes, include_top=True):
    base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    x = L.GlobalAveragePooling2D()(base.output)
    if include_top:
        x = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, x, name="MobileNetV2")

def get_vit_small(input_shape, num_classes, include_top=True):
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
    if include_top:
        x = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, x, name="ViT_Small")

def window_attention(x, window_size, num_heads, projection_dim, name=None):
    B = tf.shape(x)[0]
    H, W = x.shape[1], x.shape[2]
    C = x.shape[3]
    
    # Partition into non-overlapping windows
    x_part = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    x_part = tf.transpose(x_part, [0, 1, 3, 2, 4, 5])
    x_part = tf.reshape(x_part, [-1, window_size * window_size, C])
    
    # Attention inside windows
    attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, name=name)(x_part, x_part)
    
    # Reverse partition
    attn = tf.reshape(attn, [-1, H // window_size, W // window_size, window_size, window_size, C])
    attn = tf.transpose(attn, [0, 1, 3, 2, 4, 5])
    attn = tf.reshape(attn, [-1, H, W, C])
    return tf.ensure_shape(attn, [None, H, W, C])

def swin_block(x, projection_dim, num_heads, window_size=7, shift_size=0, name=""):
    x1 = L.LayerNormalization(epsilon=1e-6)(x)
    
    if shift_size > 0:
        x1 = tf.roll(x1, shift=[-shift_size, -shift_size], axis=[1, 2])
        
    attn = window_attention(x1, window_size, num_heads, projection_dim, name=name+"_attn")
    
    if shift_size > 0:
        attn = tf.roll(attn, shift=[shift_size, shift_size], axis=[1, 2])
        
    x2 = L.Add()([attn, x])
    x3 = L.LayerNormalization(epsilon=1e-6)(x2)
    x3 = L.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
    x3 = L.Dropout(0.1)(x3)
    x3 = L.Dense(projection_dim)(x3)
    x3 = L.Dropout(0.1)(x3)
    return L.Add()([x3, x2])

def get_swin_tiny(input_shape, num_classes, include_top=True):
    # True Hierarchical Transformer (Swin) with ViT-like hyperparams
    inputs = L.Input(shape=input_shape)
    
    # Stage 1: 56x56
    x = L.Conv2D(128, kernel_size=4, strides=4, padding="valid")(inputs)
    x = swin_block(x, 128, 4, window_size=7, shift_size=0, name="swin_st1")
    
    # Stage 2: 28x28
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.Conv2D(128, kernel_size=2, strides=2, padding="valid")(x)
    x = swin_block(x, 128, 4, window_size=7, shift_size=3, name="swin_st2")
    
    # Stage 3: 14x14
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.Conv2D(128, kernel_size=2, strides=2, padding="valid")(x)
    x = swin_block(x, 128, 4, window_size=7, shift_size=0, name="swin_st3")
    
    # Stage 4: 7x7
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.Conv2D(128, kernel_size=2, strides=2, padding="valid")(x)
    x = swin_block(x, 128, 4, window_size=7, shift_size=3, name="swin_st4")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    if include_top:
        x = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, x, name="SwinTiny_Custom")

def get_cvt_model(input_shape, num_classes, include_top=True):
    inputs = L.Input(shape=input_shape)
    # stride=16 produces 14x14=196 tokens (matching ViT speed)
    # 128-dim, 4 blocks, 4 heads — lighter than Phase 2 (6 blocks, 8 heads)
    x = conv_embedding(inputs, filters=128, kernel_size=7, strides=16)
    x = transformer_block(x, projection_dim=128, num_heads=4, name="cvt_attn_0")
    x = transformer_block(x, projection_dim=128, num_heads=4, name="cvt_attn_1")
    x = transformer_block(x, projection_dim=128, num_heads=4, name="cvt_attn_2")
    x = transformer_block(x, projection_dim=128, num_heads=4, name="cvt_attn_3")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    if include_top:
        x = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, x, name="CvT_Custom")

def get_cvt_model_medium(input_shape, num_classes, include_top=True):
    inputs = L.Input(shape=input_shape)
    # Phase 2 with RoPE + CPE — 192-dim, 4 heads, 6 layers
    x = conv_embedding(inputs, filters=192, kernel_size=7, strides=16)
    
    grid_h = input_shape[0] // 16
    grid_w = input_shape[1] // 16
    
    for i in range(6):
        x = rope_transformer_block(x, projection_dim=192, num_heads=4, grid_h=grid_h, grid_w=grid_w, name=f"cvt_med_rope_{i}")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    
    if include_top:
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.3)(x)
        x = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, x, name="CvT_Custom_Medium")

def get_cvt_model_classic_medium(input_shape, num_classes, include_top=True):
    inputs = L.Input(shape=input_shape)
    # Single Phase Classic — 192-dim, 4 heads, 6 layers
    x = conv_embedding(inputs, filters=192, kernel_size=7, strides=16)
    
    # Additive positional embeddings instead of RoPE
    num_patches = (input_shape[0] // 16) * (input_shape[1] // 16)
    pos_embed = L.Embedding(input_dim=num_patches, output_dim=192)(tf.range(start=0, limit=num_patches, delta=1))
    x = x + pos_embed
    
    for i in range(6):
        x = classic_transformer_block(x, projection_dim=192, num_heads=4, name=f"cvt_med_classic_{i}")
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    
    if include_top:
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.3)(x)
        x = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, x, name="CvT_Custom_Classic_Medium")

def get_cvt_model_large(input_shape, num_classes, include_top=True):
    # For backward compatibility, map this to the exact same medium architectur
    return get_cvt_model_medium(input_shape, num_classes, include_top)

def get_conformer_model_medium(input_shape, num_classes, include_top=True):
    """
    SuperConformer (Multi-Scale CvT backbone with Local Bypass).
    """
    inputs = L.Input(shape=input_shape)
    
    # Branch A (Fine Texture): 8x8 patches (e.g. 28x28 grid) -> pooled to 14x14
    grid_h_fine, grid_w_fine = input_shape[0] // 8, input_shape[1] // 8
    branch_fine = conv_embedding(inputs, filters=96, kernel_size=3, strides=8, name="branch_fine")
    branch_fine_reshaped = L.Reshape((grid_h_fine, grid_w_fine, 96))(branch_fine)
    branch_fine_pooled = L.Conv2D(96, kernel_size=3, strides=2, padding='same', name="branch_fine_pool")(branch_fine_reshaped)
    branch_fine = L.Reshape((input_shape[0]//16 * input_shape[1]//16, 96))(branch_fine_pooled)
    
    # Branch B (Coarse Structure): 16x16 patches (14x14 grid)
    branch_coarse = conv_embedding(inputs, filters=96, kernel_size=7, strides=16, name="branch_coarse")
    
    # Early Fusion
    x = L.Concatenate(axis=-1)([branch_fine, branch_coarse]) # 192-dim
    local_bypass = x
    
    grid_h = input_shape[0] // 16
    grid_w = input_shape[1] // 16
    
    # Enhanced Local Bias: cpe_kernel=5
    for i in range(6):
        x = rope_transformer_block(x, projection_dim=192, num_heads=4, grid_h=grid_h, grid_w=grid_w, name=f"conf_rope_{i}", cpe_kernel=5)
    
    x = L.LayerNormalization(epsilon=1e-6)(x)
    
    # Bypass texture extraction
    bypass_shape = (grid_h, grid_w, 192)
    bypass = L.Reshape(bypass_shape)(local_bypass)
    bypass = L.SeparableConv2D(192, 3, padding='same', activation='swish')(bypass)
    bypass = L.Reshape((grid_h * grid_w, 192))(bypass)
    
    if include_top:
        x = L.Add()([x, bypass])
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.3)(x)
        outputs = L.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs, name="SuperConformer_Medium")
    else:
        return tf.keras.Model(inputs, [x, bypass], name="SuperConformer_Base")

def get_model(model_name, input_shape, num_classes, include_top=True):
    if model_name == "ResNet50":
        return get_resnet50(input_shape, num_classes, include_top=include_top)
    elif model_name == "EfficientNetB0":
        return get_efficientnet_b0(input_shape, num_classes, include_top=include_top)
    elif model_name == "MobileNetV2":
        return get_mobilenet_v2(input_shape, num_classes, include_top=include_top)
    elif model_name == "ViT":
        return get_vit_small(input_shape, num_classes, include_top=include_top)
    elif model_name == "SwinTiny":
        return get_swin_tiny(input_shape, num_classes, include_top=include_top)
    elif model_name == "CvT":
        # Global/Single phase now uses the Classic Medium model
        return get_cvt_model_classic_medium(input_shape, num_classes, include_top=include_top)
    elif model_name == "Conformer":
        return get_conformer_model_medium(input_shape, num_classes, include_top=include_top)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_dual_input_model(model_name, input_shape, num_crops, num_diseases):
    """
    Builds a single specialist model that takes both an image and a crop label.
    """
    image_input = L.Input(shape=input_shape, name="image_input")
    crop_input = L.Input(shape=(num_crops,), name="crop_input")

    # Get image features (without final classification layer)
    if model_name == "CvT":
        base_model = get_cvt_model_medium(input_shape, num_classes=None, include_top=False)
        image_features = base_model(image_input)
    elif model_name == "Conformer":
        base_model = get_conformer_model_medium(input_shape, num_classes=None, include_top=False)
        transformer_features, bypass_features = base_model(image_input)
    else:
        base_model = get_model(model_name, input_shape, num_classes=None, include_top=False)
        image_features = base_model(image_input)

    # Embed crop input
    crop_features = L.Dense(192, activation='relu')(crop_input)
    crop_features = L.LayerNormalization(epsilon=1e-6)(crop_features)

    if model_name == "Conformer":
        # Dynamic Cross-Attention Fusion (SE block is explicitly removed)
        augmented_image_features = L.Add(name="augmented_features_add")([transformer_features, bypass_features])
        
        # crop_query acts to dynamically slice the most relevant diagnostic features from the image
        crop_query = L.Reshape((1, 192), name="crop_reshape_1")(crop_features)
        
        # Native Cross Attention - Query is Crop, Key/Value is Image Sequence
        attention_out = L.MultiHeadAttention(num_heads=4, key_dim=192, name="crop_conditioned_attention")(
            query=crop_query, value=augmented_image_features, key=augmented_image_features
        )
        
        # 1. Decoder MLP Block (adds required architectural non-linearity)
        attn_out_norm = L.LayerNormalization(epsilon=1e-6, name="decoder_ln")(attention_out)
        mlp_out = L.Dense(384, activation=tf.nn.gelu, name="decoder_mlp_1")(attn_out_norm)
        mlp_out = L.Dropout(0.1, name="decoder_mlp_dropout")(mlp_out)
        mlp_out = L.Dense(192, name="decoder_mlp_2")(mlp_out)
        attention_out = L.Add(name="decoder_residual_add")([attention_out, mlp_out])
        
        # attention_out is already (Batch, 1, 192), extracting pure crop-contextualized image patterns
        merged = L.Reshape((192,), name="reshape_merged")(attention_out)
        
        # 2. Bypass Gradient Superhighway
        # Funnels high-freq texture natively to the decision layer 
        pure_bypass = L.GlobalAveragePooling1D(name="pure_bypass_gap")(bypass_features)
        merged = L.Add(name="superhighway_add")([merged, pure_bypass])
        
        merged = L.Dropout(0.3, name="final_dropout")(merged)

    elif model_name == "CvT":
        # PROMPT FUSION: Treat the crop embedding as a 'Prompt Token'
        # Reshape to sequence of length 1: (batch, 1, 192)
        crop_token = L.Reshape((1, 192))(crop_features)
        
        # Concatenate the crop token to the image token sequence (batch, 196+1, 192)
        merged_sequence = L.Concatenate(axis=1)([crop_token, image_features])
        
        # Cross-Attention: The transformer self-attention naturally lets image patches 
        # pull context dynamically from the crop token.
        merged_sequence = classic_transformer_block(merged_sequence, projection_dim=192, num_heads=4, name="prompt_cross_fusion")
        
        # Pool all spatially-contextualized tokens back into a vector
        merged = L.GlobalAveragePooling1D()(merged_sequence)
        merged = L.Dropout(0.3)(merged)
    else:
        # Standard CNN flat fusion
        merged = L.Concatenate()([image_features, crop_features])
        merged = L.Dense(256, activation='relu')(merged)
        merged = L.Dropout(0.3)(merged)

    outputs = L.Dense(num_diseases, activation='softmax', name="disease_output")(merged)

    return tf.keras.Model(inputs=[image_input, crop_input], outputs=outputs, name=f"{model_name}_DualInputPhase2")
