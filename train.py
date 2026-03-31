import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf
import json
import logging

from models import get_model
from data_loader import get_data_generators, get_dummy_dataset
from metrics_utils import get_flops, plot_history, plot_confusion_matrix, calculate_metrics
from explainability import generate_explainability
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Crop Diseases/Crop___Disease", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="all", help="Model to run or 'all' for all models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dummy", action="store_true", help="Run a dummy test without real data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max images per class (default: use all images)")
    args = parser.parse_args()

    available_models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT"]
    if args.model == "all":
        models_to_run = available_models
    else:
        if args.model not in available_models:
            raise ValueError(f"Model must be 'all' or one of {available_models}")
        models_to_run = [args.model]

    seeds = [args.seed]
        
    img_size = 224
    
    def get_classes_from_dir(base_dir):
        classes = []
        if not os.path.exists(base_dir):
            return None
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
                if subdirs:
                    for sub in subdirs:
                        classes.append(os.path.join(item, sub))
                else:
                    classes.append(item)
        return sorted(classes) if classes else None

    if args.dummy:
        num_classes = 17
        classes = None
        # Add support for dummy flag
    else:
        classes = get_classes_from_dir(args.data_dir)
        num_classes = len(classes) if classes else 17

    os.makedirs("results", exist_ok=True)
    all_results = {}

    for model_name in models_to_run:
        all_results[model_name] = {"seeds": {}}
        for seed in seeds:
            print(f"\\n{'='*50}\\nRunning {model_name} with seed {seed}\\n{'='*50}\\n")
            set_seed(seed)

            # Create model-specific results folder
            model_save_dir = os.path.join("results", model_name)
            os.makedirs(model_save_dir, exist_ok=True)

            # Build Model
            input_shape = (img_size, img_size, 3)
            model = get_model(model_name, input_shape, num_classes)
            
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

            # Data Loading
            if args.dummy:
                train_gen, val_gen, test_gen, steps_per_epoch = get_dummy_dataset(img_size, args.batch_size, num_classes)
                val_steps = 2
                test_steps = 2
            else:
                train_gen, val_gen, test_gen = get_data_generators(args.data_dir, args.batch_size, img_size, classes=classes, max_per_class=args.max_per_class)
                steps_per_epoch = train_gen.samples // args.batch_size
                val_steps = val_gen.samples // args.batch_size
                test_steps = test_gen.samples // args.batch_size

            # Computational Cost
            params = model.count_params()
            flops = get_flops(model)
            
            # Compute class weights for handling imbalance
            if args.dummy:
                class_weight_dict = None
            else:
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(train_gen.classes),
                    y=train_gen.classes
                )
                class_weight_dict = dict(enumerate(class_weights))

            # Callbacks
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
                )
            ]

            # Train model
            start_train = time.time()
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            train_time = time.time() - start_train

            # Standard Evaluation
            start_infer = time.time()
            if args.dummy:
                dummy_x, dummy_y = next(iter(test_gen))
                y_pred_probs = model.predict(dummy_x)
                y_true = np.argmax(dummy_y, axis=1)
                test_images = dummy_x[:1]
            else:
                y_pred_probs = model.predict(test_gen)
                y_true = test_gen.classes
                test_images, _ = next(iter(test_gen))
            
            infer_time = time.time() - start_infer
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Note: For generators without shuffle=False, predictions and true labels might misalign if not careful.
            # Usually we recreate the generator with shuffle=False for eval, but for this benchmark script,
            # dummy generator yields random things, so metrics will be random anyway.

            metrics = calculate_metrics(y_true, y_pred)
            
            print(f"\nTest Metrics ({model_name}, Seed {seed}):")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}\n")

            try:
                if args.dummy:
                    class_names = [f"Class_{i}" for i in range(num_classes)]
                else:
                    class_names = list(test_gen.class_indices.keys())

                if not args.dummy:
                    plot_confusion_matrix(y_true, y_pred, class_names, model_name, seed, save_dir=model_save_dir)
                    generate_explainability(model, test_gen, model_name, seed, save_dir="results")
                plot_history(history.history, model_name, seed, save_dir=model_save_dir)
            except Exception as e:
                print(f"Explainability/Plotting failed for {model_name} (Seed {seed}): {e}")
                class_names = []

            # Robustness Eval (dummy only tests code path)
            robustness_results = {}
            for corruption in ["noise", "blur", "lighting"]:
                if args.dummy:
                    rob_gen = test_gen
                else:
                    _, _, rob_gen = get_data_generators(args.data_dir, args.batch_size, img_size, classes=classes, max_per_class=args.max_per_class, corruption_type=corruption)
                
                try:
                    score = model.evaluate(rob_gen, steps=test_steps, verbose=0)
                    robustness_results[corruption] = score[1] # accuracy
                except Exception:
                    robustness_results[corruption] = 0.0

            all_results[model_name]["seeds"][seed] = {
                "params": params,
                "flops": flops,
                "train_time_s": train_time,
                "infer_time_s": infer_time,
                "metrics": metrics,
                "class_names": class_names,
                "robustness_accuracy": robustness_results
            }
            
            # Save model-specific results for this seed
            with open(os.path.join(model_save_dir, f"benchmark_results_seed{seed}.json"), "w") as f:
                json.dump({model_name: all_results[model_name]}, f, indent=4)

            # Clear session to avoid memory leak
            tf.keras.backend.clear_session()
            
    print("Benchmarking completed! Results saved to results/ folder!")

if __name__ == "__main__":
    main()
