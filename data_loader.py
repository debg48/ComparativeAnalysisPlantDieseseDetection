import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

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

def get_data_generators(data_path, batch_size, img_size, corruption_type=None):
    """
    Returns train and val generators. If data_path doesn't exist, it can be handled outside.
    """
    def preprocessing_function(image):
        return custom_preprocessing(image, corruption_type)

    if corruption_type is None:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=30,
            zoom_range=0.3,
            horizontal_flip=True
        )
        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    else:
        # Robust generators
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=0.2
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=0.2
        )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = val_datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen

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
    # Return (dataset, dataset, steps_per_epoch)
    return dataset, dataset, 10
