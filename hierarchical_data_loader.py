import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def discover_hierarchy(data_path):
    """
    Discover the crop → disease hierarchy from the nested folder structure.
    Returns:
        crop_to_diseases: dict mapping crop name → list of disease folder relative paths
        all_classes: sorted list of all relative class paths (e.g. 'Corn/Corn___Common_Rust')
    """
    crop_to_diseases = {}
    all_classes = []

    for crop in sorted(os.listdir(data_path)):
        crop_path = os.path.join(data_path, crop)
        if not os.path.isdir(crop_path):
            continue
        diseases = []
        for disease in sorted(os.listdir(crop_path)):
            disease_path = os.path.join(crop_path, disease)
            if os.path.isdir(disease_path):
                rel_path = os.path.join(crop, disease)
                diseases.append(rel_path)
                all_classes.append(rel_path)
        if diseases:
            crop_to_diseases[crop] = diseases

    return crop_to_diseases, sorted(all_classes)


def build_master_dataframe(data_path, all_classes):
    """
    Build a single DataFrame of all images with columns: filename, disease_label, crop_label.
    """
    rows = []
    for rel_class_path in all_classes:
        full_path = os.path.join(data_path, rel_class_path)
        crop_name = rel_class_path.split(os.sep)[0]
        if os.path.isdir(full_path):
            for f in os.listdir(full_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rows.append({
                        'filename': os.path.join(rel_class_path, f),
                        'disease_label': rel_class_path,
                        'crop_label': crop_name
                    })

    return pd.DataFrame(rows)


def split_dataframe(df, random_state=42):
    """
    Split into 70% train / 15% val / 15% test with stratification on disease_label.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df['disease_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_df['disease_label']
    )
    return train_df, val_df, test_df


def _make_generator(datagen, df, data_path, label_col, img_size, batch_size, shuffle, classes=None):
    """Helper to create a flow_from_dataframe generator."""
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=data_path,
        x_col="filename",
        y_col=label_col,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle,
        classes=classes
    )


def get_crop_generators(data_path, train_df, val_df, test_df, img_size, batch_size):
    """
    Phase 1: Crop-level generators (5-class: Corn, Potato, Rice, Wheat, sugarcane).
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = _make_generator(train_datagen, train_df, data_path, "crop_label", img_size, batch_size, shuffle=True)
    val_gen = _make_generator(val_test_datagen, val_df, data_path, "crop_label", img_size, batch_size, shuffle=False)
    test_gen = _make_generator(val_test_datagen, test_df, data_path, "crop_label", img_size, batch_size, shuffle=False)

    return train_gen, val_gen, test_gen


def get_disease_generators_for_crop(data_path, train_df, val_df, test_df, crop_name, img_size, batch_size, all_classes=None):
    """
    Phase 2: Per-crop disease generators.
    Filters the master split to only include images from the given crop.
    """
    crop_train = train_df[train_df['crop_label'] == crop_name].copy()
    crop_val = val_df[val_df['crop_label'] == crop_name].copy()
    crop_test = test_df[test_df['crop_label'] == crop_name].copy()

    if len(crop_train) == 0 or len(crop_val) == 0 or len(crop_test) == 0:
        print(f"  [WARNING] Not enough data for crop '{crop_name}', skipping.")
        return None, None, None

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = _make_generator(train_datagen, crop_train, data_path, "disease_label", img_size, batch_size, shuffle=True, classes=all_classes)
    val_gen = _make_generator(val_test_datagen, crop_val, data_path, "disease_label", img_size, batch_size, shuffle=False, classes=all_classes)
    test_gen = _make_generator(val_test_datagen, crop_test, data_path, "disease_label", img_size, batch_size, shuffle=False, classes=all_classes)

    return train_gen, val_gen, test_gen

class DualInputWrapper(tf.keras.utils.Sequence):
    """
    Wraps an ImageDataGenerator to output `((X, y_crop), y_disease)`.
    """
    def __init__(self, df, data_path, img_size, batch_size, datagen, crop_to_idx, disease_to_idx, shuffle):
        self.df = df.copy()
        # Use a dummy column ID so the internal generator returns the row indices
        self.df['__id'] = np.arange(len(self.df))
        
        self.gen = datagen.flow_from_dataframe(
            dataframe=self.df,
            directory=data_path,
            x_col="filename",
            y_col="__id",
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=shuffle
        )
        self.num_crops = len(crop_to_idx)
        self.num_diseases = len(disease_to_idx)
        
        # Pre-compute labels for fast lookup
        self.crop_labels = np.array([crop_to_idx[c] for c in self.df['crop_label']])
        self.disease_labels = np.array([disease_to_idx[d] for d in self.df['disease_label']])
        
        # To make it compatible with history plotting scripts that check .classes
        self.classes = self.disease_labels
        self.class_indices = disease_to_idx
        self.samples = len(self.df)
        
    def __len__(self):
        return len(self.gen)
        
    def __getitem__(self, i):
        X, ids = self.gen[i]
        ids = ids.astype(int)
        
        c_ids = self.crop_labels[ids]
        d_ids = self.disease_labels[ids]
        
        y_crop = tf.keras.utils.to_categorical(c_ids, num_classes=self.num_crops)
        y_disease = tf.keras.utils.to_categorical(d_ids, num_classes=self.num_diseases)
        
        return (X, y_crop), y_disease
        
    def on_epoch_end(self):
        self.gen.on_epoch_end()


def get_dual_input_generators(data_path, train_df, val_df, test_df, img_size, batch_size, crop_to_idx, disease_to_idx):
    """
    Returns dual-input generator wrappers for training a single Phase 2 specialist.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = DualInputWrapper(
        train_df, data_path, img_size, batch_size, train_datagen, crop_to_idx, disease_to_idx, shuffle=True
    )
    val_gen = DualInputWrapper(
        val_df, data_path, img_size, batch_size, val_test_datagen, crop_to_idx, disease_to_idx, shuffle=False
    )
    test_gen = DualInputWrapper(
        test_df, data_path, img_size, batch_size, val_test_datagen, crop_to_idx, disease_to_idx, shuffle=False
    )

    return train_gen, val_gen, test_gen
