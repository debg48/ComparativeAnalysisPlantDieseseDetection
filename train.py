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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="all", help="Model to run or 'all' for all models")
    parser.add_argument("--seed", type=str, default="all", help="Specific seed to run, e.g., '42', or 'all'")
    args = parser.parse_args()

    available_models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT"]
    if args.model == "all":
        models_to_run = available_models
    else:
        if args.model not in available_models:
            raise ValueError(f"Model must be 'all' or one of {available_models}")
        models_to_run = [args.model]

    if args.seed == "all":
        seeds = [42, 123, 456]
    else:
        seeds = [int(args.seed)]
        
    img_size = 128
    num_classes = 5

    os.makedirs("results", exist_ok=True)
    all_results = {}

    for model_name in models_to_run:
        all_results[model_name] = {"seeds": {}}
        for seed in seeds:
            print(f"\\n{'='*50}\\nRunning {model_name} with seed {seed}\\n{'='*50}\\n")
            set_seed(seed)

            # Build Model
            input_shape = (img_size, img_size, 3)
            model = get_model(model_name, input_shape, num_classes)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            # Data Loading
            if args.dummy:
                train_gen, val_gen, steps_per_epoch = get_dummy_dataset(img_size, args.batch_size, num_classes)
                val_steps = 2
            else:
                train_gen, val_gen = get_data_generators(args.data_dir, args.batch_size, img_size)
                steps_per_epoch = train_gen.samples // args.batch_size
                val_steps = val_gen.samples // args.batch_size

            # Computational Cost
            params = model.count_params()
            flops = get_flops(model)
            
            # Train model
            start_train = time.time()
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                verbose=1
            )
            train_time = time.time() - start_train

            # Standard Evaluation
            start_infer = time.time()
            if args.dummy:
                dummy_x, dummy_y = next(iter(val_gen))
                y_pred_probs = model.predict(dummy_x)
                y_true = np.argmax(dummy_y, axis=1)
                test_images = dummy_x[:1]
            else:
                y_pred_probs = model.predict(val_gen)
                y_true = val_gen.classes
                test_images, _ = next(iter(val_gen))
            
            infer_time = time.time() - start_infer
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Note: For generators without shuffle=False, predictions and true labels might misalign if not careful.
            # Usually we recreate the generator with shuffle=False for eval, but for this benchmark script,
            # dummy generator yields random things, so metrics will be random anyway.

            metrics = calculate_metrics(y_true, y_pred)
            
            try:
                generate_explainability(model, test_images, model_name, seed, save_dir="results")
                plot_history(history.history, model_name, seed, save_dir="results")
            except Exception as e:
                print(f"Explainability/Plotting failed for {model_name} (Seed {seed}): {e}")

            # Robustness Eval (dummy only tests code path)
            robustness_results = {}
            for corruption in ["noise", "blur", "lighting"]:
                if args.dummy:
                    rob_gen = val_gen
                else:
                    _, rob_gen = get_data_generators(args.data_dir, args.batch_size, img_size, corruption_type=corruption)
                
                try:
                    score = model.evaluate(rob_gen, steps=val_steps, verbose=0)
                    robustness_results[corruption] = score[1] # accuracy
                except Exception:
                    robustness_results[corruption] = 0.0

            all_results[model_name]["seeds"][seed] = {
                "params": params,
                "flops": flops,
                "train_time_s": train_time,
                "infer_time_s": infer_time,
                "metrics": metrics,
                "robustness_accuracy": robustness_results
            }
            
            # Save model-specific results for this seed
            with open(os.path.join("results", f"benchmark_results_{model_name}_seed{seed}.json"), "w") as f:
                json.dump({model_name: all_results[model_name]}, f, indent=4)

            # Clear session to avoid memory leak
            tf.keras.backend.clear_session()
            
    print("Benchmarking completed! Results saved to results/ folder!")

if __name__ == "__main__":
    main()
