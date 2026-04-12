import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def add_noise(image):
    noise = np.random.normal(0, 0.05 * 255, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

def add_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def change_lighting(image):
    # Randomly change brightness
    value = np.random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

def custom_preprocessing(image, corruption_type=None):
    if corruption_type == "noise":
        image = add_noise(image)
    if corruption_type == "blur":
        image = add_blur(image)
    if corruption_type == "lighting":
        image = change_lighting(image)
        
    # Scale to 0-1
    return image / 255.0

def get_data_generators(data_path, batch_size, img_size, classes=None, max_per_class=None, corruption_type=None):
    """
    Returns train, validation, and test generators as 70/15/15 disjoint sets.
    """
    # 1. Collect all images and their labels
    file_paths = []
    labels = []
    
    # If classes is provided, it contains relative paths like 'Corn/Corn___Common_Rust'
    for rel_class_path in classes:
        full_class_path = os.path.join(data_path, rel_class_path)
        if os.path.isdir(full_class_path):
            files = [os.path.join(rel_class_path, f) for f in os.listdir(full_class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            file_paths.extend(files)
            labels.extend([rel_class_path] * len(files))
            
    df = pd.DataFrame({'filename': file_paths, 'label': labels})
    
    # Subsample per class if max_per_class is set
    if max_per_class is not None:
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_per_class), random_state=42)
        ).reset_index(drop=True)
        print(f"[Subsampled] {len(df)} total images (max {max_per_class} per class)")
    
    # 2. Split into Train (70%), Validation (15%), and Test (15%)
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    
    # Second split: split temp 50/50 to get 15% val and 15% test
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    
    def preprocessing_function(image):
        return custom_preprocessing(image, corruption_type)

    # 3. Define Generators
    if corruption_type is None:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        val_test_datagen = ImageDataGenerator(rescale=1./255)
    else:
        # Robust generators
        train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        val_test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    # 4. Create Flow from Dataframes
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=data_path,
        x_col="filename",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=data_path,
        x_col="filename",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=data_path,
        x_col="filename",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def get_dummy_dataset(img_size, batch_size, num_classes):
    """Generates a dummy dataset for testing if data folder is empty."""
    def generator():
        while True:
            images = np.random.rand(batch_size, img_size, img_size, 3).astype(np.float32)
            labels = np.zeros((batch_size, num_classes), dtype=np.float32)
            for i in range(batch_size):
                labels[i, np.random.randint(0, num_classes)] = 1
            yield images, labels
            
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32)
        )
    )
    # Return (train, val, test, steps_per_epoch)
    return dataset, dataset, dataset, 10
