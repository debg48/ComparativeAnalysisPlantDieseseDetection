import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf
import json

from models import get_model, get_dual_input_model
from hierarchical_data_loader import (
    discover_hierarchy, build_master_dataframe, split_dataframe,
    get_crop_generators, get_disease_generators_for_crop,
    get_dual_input_generators
)
from metrics_utils import get_flops, plot_history, plot_confusion_matrix, calculate_metrics
from explainability import generate_explainability


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_and_evaluate(model, train_gen, val_gen, test_gen, epochs, batch_size, model_name_tag, callbacks=None, verbose=1):
    """Train a model and return metrics + timing."""
    steps_per_epoch = max(1, train_gen.samples // batch_size)
    val_steps = max(1, val_gen.samples // batch_size)
    test_steps = max(1, test_gen.samples // batch_size)

    # Computational cost
    params = model.count_params()
    flops = get_flops(model)

    # Train
    start_train = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose
    )
    train_time = time.time() - start_train

    # Evaluate on test set
    start_infer = time.time()
    y_pred_probs = model.predict(test_gen)
    y_true = test_gen.classes
    infer_time = time.time() - start_infer

    y_pred = np.argmax(y_pred_probs, axis=1)
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "params": params,
        "flops": flops,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "metrics": metrics,
        "history": history.history,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs
    }


def end_to_end_evaluate(crop_model, phase2_model, test_df, data_path,
                         crop_to_diseases, img_size, batch_size):
    """
    Run full two-stage inference on the test set.
    Returns overall metrics + error propagation analysis.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create a test generator with disease labels (ground truth)
    test_gen_disease = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=data_path,
        x_col="filename",
        y_col="disease_label",
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    # Create a test generator with crop labels (for Phase 1 ground truth)
    test_gen_crop = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=data_path,
        x_col="filename",
        y_col="crop_label",
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    crop_idx_to_name = {v: k for k, v in test_gen_crop.class_indices.items()}
    disease_idx_to_name = {v: k for k, v in test_gen_disease.class_indices.items()}

    # Build a unified disease label mapping
    all_disease_names = sorted(test_gen_disease.class_indices.keys())

    total = len(test_df)
    correct_routing = 0
    incorrect_routing = 0
    correct_when_routed_right = 0
    correct_when_routed_wrong = 0
    total_routed_right = 0
    total_routed_wrong = 0

    final_preds = []
    true_labels = []

    start_time = time.time()

    for i in range(total):
        img, disease_onehot = test_gen_disease[i]
        _, crop_onehot = test_gen_crop[i]

        true_disease_idx = np.argmax(disease_onehot[0])
        true_crop_idx = np.argmax(crop_onehot[0])
        true_crop_name = crop_idx_to_name[true_crop_idx]
        true_disease_name = disease_idx_to_name[true_disease_idx]

        # Phase 1: Predict crop
        crop_pred_probs = crop_model.predict(img, verbose=0)
        pred_crop_idx = np.argmax(crop_pred_probs[0])
        pred_crop_name = crop_idx_to_name[pred_crop_idx]

        routing_correct = (pred_crop_name == true_crop_name)

        # Phase 2: Route to joint specialist with predicted crop feature
        pred_crop_onehot = np.zeros((1, len(crop_idx_to_name)))
        pred_crop_onehot[0, pred_crop_idx] = 1.0

        disease_pred_probs = phase2_model.predict([img, pred_crop_onehot], verbose=0)[0]
        
        # Mask probabilities to enforce hierarchical constraint
        valid_diseases = set(crop_to_diseases[pred_crop_name])
        for idx, d_name in enumerate(phase2_model._specialist_classes):
            if d_name not in valid_diseases:
                disease_pred_probs[idx] = 0.0
                
        pred_disease_local_idx = np.argmax(disease_pred_probs)
        pred_disease_name = phase2_model._specialist_classes[pred_disease_local_idx]

        # Map back to global disease index
        pred_disease_global_idx = all_disease_names.index(pred_disease_name) if pred_disease_name in all_disease_names else -1
        final_preds.append(pred_disease_global_idx)
        true_labels.append(true_disease_idx)

        # Error propagation tracking
        final_correct = (pred_disease_name == true_disease_name)
        if routing_correct:
            total_routed_right += 1
            if final_correct:
                correct_when_routed_right += 1
        else:
            total_routed_wrong += 1
            if final_correct:
                correct_when_routed_wrong += 1

    total_infer_time = time.time() - start_time

    # Convert to arrays
    final_preds = np.array(final_preds)
    true_labels = np.array(true_labels)

    overall_metrics = calculate_metrics(true_labels, final_preds)

    error_propagation = {
        "total_samples": total,
        "correctly_routed": total_routed_right,
        "incorrectly_routed": total_routed_wrong,
        "routing_accuracy": total_routed_right / total if total > 0 else 0,
        "accuracy_when_routed_correctly": correct_when_routed_right / total_routed_right if total_routed_right > 0 else 0,
        "accuracy_when_routed_incorrectly": correct_when_routed_wrong / total_routed_wrong if total_routed_wrong > 0 else 0,
    }

    return {
        "overall_metrics": overall_metrics,
        "error_propagation": error_propagation,
        "total_infer_time_s": total_infer_time,
        "y_true": true_labels,
        "y_pred": final_preds,
        "disease_names": all_disease_names
    }


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Hierarchical Crop Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="data/Crop Diseases/Crop___Disease", help="Path to dataset")
    parser.add_argument("--p1_epochs", type=int, default=15, help="Number of training epochs for Phase 1")
    parser.add_argument("--p2_epochs", type=int, default=20, help="Number of training epochs for Phase 2")
    parser.add_argument("--model", type=str, default="CvT", help="Model architecture to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    available_models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT", "Conformer"]
    if args.model not in available_models:
        raise ValueError(f"Model must be one of {available_models}")

    seeds = [args.seed]

    img_size = 224

    # Discover hierarchy
    crop_to_diseases, all_classes = discover_hierarchy(args.data_dir)
    crops = sorted(crop_to_diseases.keys())
    num_crops = len(crops)

    print(f"\n{'='*60}")
    print(f"HIERARCHICAL CLASSIFICATION PIPELINE")
    print(f"Architecture: {args.model}")
    print(f"Crops: {crops}")
    print(f"Total disease classes: {len(all_classes)}")
    for crop, diseases in crop_to_diseases.items():
        print(f"  {crop}: {len(diseases)} classes")
    print(f"{'='*60}\n")

    # Build master dataframe (use ALL images — no cap)
    master_df = build_master_dataframe(args.data_dir, all_classes)
    print(f"Total images: {len(master_df)}")

    # Create result directory
    model_hier_dir = os.path.join("results", f"{args.model}_Hierarchical")
    os.makedirs(model_hier_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*50}\nSeed: {seed}\n{'='*50}\n")
        set_seed(seed)

        # Split (same split for all stages)
        train_df, val_df, test_df = split_dataframe(master_df, random_state=seed)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        seed_results = {}
        seed_dir = os.path.join(model_hier_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # =============================================
        # PHASE 1: Train Crop Router
        # =============================================
        print(f"\n--- Phase 1: Training Crop Router ({num_crops} classes) ---\n")

        crop_train_gen, crop_val_gen, crop_test_gen = get_crop_generators(
            args.data_dir, train_df, val_df, test_df, img_size, args.batch_size
        )

        input_shape = (img_size, img_size, 3)
        if args.model in ["CvT", "Conformer"]:
            # Phase 1: Use CNN router for high routing accuracy on the simple 5-class crop task
            from models import get_cnn_router
            crop_model = get_cnn_router(input_shape, num_crops)
        else:
            crop_model = get_model(args.model, input_shape, num_crops)
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
        crop_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Phase 1 callbacks
        p1_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
            )
        ]

        crop_results = train_and_evaluate(
            crop_model, crop_train_gen, crop_val_gen, crop_test_gen,
            args.p1_epochs, args.batch_size, f"Phase1_CropRouter",
            callbacks=p1_callbacks
        )

        # Save Phase 1 plots and model
        phase1_dir = os.path.join(seed_dir, "phase1_crop_router")
        os.makedirs(phase1_dir, exist_ok=True)
        plot_history(crop_results["history"], f"{args.model}_CropRouter", seed, save_dir=phase1_dir)
        crop_model.save(os.path.join(phase1_dir, "crop_router_model.h5"))
        print(f"  Phase 1 model saved to: {phase1_dir}/crop_router_model.h5")

        crop_class_names = list(crop_test_gen.class_indices.keys())
        
        m = crop_results["metrics"]
        print(f"\nPhase 1 (Crop Router) Metrics (Seed {seed}):")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1_score']:.4f}\n")

        plot_confusion_matrix(crop_results["y_true"], crop_results["y_pred"],
                              crop_class_names, f"{args.model}_CropRouter", seed, save_dir=phase1_dir)

        seed_results["phase1_crop_router"] = {
            "params": crop_results["params"],
            "flops": crop_results["flops"],
            "train_time_s": crop_results["train_time_s"],
            "infer_time_s": crop_results["infer_time_s"],
            "metrics": crop_results["metrics"],
            "class_names": crop_class_names
        }

        print(f"\n  Phase 1 Accuracy: {crop_results['metrics']['accuracy']:.4f}\n")

        # =============================================
        # PHASE 2: Train Joint Specialist
        # =============================================
        print(f"\n--- Phase 2: Training Dual-Input Joint Specialist ({len(all_classes)} classes) ---\n")

        crop_to_idx = crop_test_gen.class_indices
        disease_to_idx = {name: i for i, name in enumerate(all_classes)}

        disease_train, disease_val, disease_test = get_dual_input_generators(
            args.data_dir, train_df, val_df, test_df, img_size, args.batch_size, crop_to_idx, disease_to_idx
        )

        tf.keras.backend.clear_session()
        set_seed(seed)

        phase2_model = get_dual_input_model(args.model, input_shape, num_crops, len(all_classes))
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
        phase2_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Phase 2 callbacks
        p2_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
            )
        ]

        spec_results = train_and_evaluate(
            phase2_model, disease_train, disease_val, disease_test,
            args.p2_epochs, args.batch_size, f"Phase2_JointSpecialist",
            callbacks=p2_callbacks
        )

        # Store class mapping on the model for end-to-end eval
        phase2_model._specialist_classes = all_classes

        # Save Phase 2 plots and model
        phase2_dir = os.path.join(seed_dir, "phase2_joint_specialist")
        os.makedirs(phase2_dir, exist_ok=True)
        plot_history(spec_results["history"], f"{args.model}_JointSpecialist", seed, save_dir=phase2_dir)
        phase2_model.save(os.path.join(phase2_dir, "joint_specialist_model.h5"))
        print(f"  Phase 2 model saved to: {phase2_dir}/joint_specialist_model.h5")

        m = spec_results["metrics"]
        print(f"\nPhase 2 (Joint Specialist) Metrics (Seed {seed}):")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1_score']:.4f}\n")

        plot_confusion_matrix(spec_results["y_true"], spec_results["y_pred"],
                              all_classes, f"{args.model}_JointSpecialist", seed, save_dir=phase2_dir)

        # Explainability for specialist (might have issues tracking inputs, wrap in try/except)
        try:
            generate_explainability(phase2_model, disease_test, f"{args.model}_JointSpecialist", seed, save_dir=seed_dir)
        except Exception as e:
            print(f"  Explainability failed for JointSpecialist: {e}")

        seed_results["phase2_specialist"] = {
            "num_classes": len(all_classes),
            "params": spec_results["params"],
            "flops": spec_results["flops"],
            "train_time_s": spec_results["train_time_s"],
            "infer_time_s": spec_results["infer_time_s"],
            "metrics": spec_results["metrics"],
            "class_names": all_classes,
            "per_crop_metrics": {}
        }
        
        print("\n  Evaluating Joint Specialist Per-Crop:")
        for crop_name in crops:
            _, _, crop_test_gen_disease = get_disease_generators_for_crop(
                args.data_dir, train_df, val_df, test_df, crop_name, img_size, args.batch_size, all_classes=all_classes
            )
            
            if crop_test_gen_disease is not None and crop_test_gen_disease.samples > 0:
                # We need to test the model with both Image and the fixed Crop label.
                # Since get_disease_generators_for_crop returned standard (x, y), we need to format it.
                crop_idx = crop_to_idx[crop_name]
                crop_onehot = np.zeros((1, num_crops))
                crop_onehot[0, crop_idx] = 1.0
                
                # To do this cleanly, use a custom generator or manual loop.
                # Manual loop over the subset is easiest for just evaluation:
                crop_test_gen_disease.reset()
                y_true = []
                y_preds = []
                steps = int(np.ceil(crop_test_gen_disease.samples / args.batch_size))
                
                for _ in range(steps):
                    batch_x, batch_y = next(crop_test_gen_disease)
                    curr_batch_size = batch_x.shape[0]
                    batch_crop = np.repeat(crop_onehot, curr_batch_size, axis=0)
                    
                    batch_preds = phase2_model.predict([batch_x, batch_crop], verbose=0)
                    y_true.extend(np.argmax(batch_y, axis=1))
                    y_preds.extend(np.argmax(batch_preds, axis=1))
                    
                crop_metrics = calculate_metrics(np.array(y_true), np.array(y_preds))
                seed_results["phase2_specialist"]["per_crop_metrics"][crop_name] = crop_metrics
                print(f"    {crop_name} Acc: {crop_metrics['accuracy']:.4f}")
            else:
                seed_results["phase2_specialist"]["per_crop_metrics"][crop_name] = None
        
        print(f"  Joint Specialist Overall Accuracy: {spec_results['metrics']['accuracy']:.4f}")

        # =============================================
        # END-TO-END EVALUATION
        # =============================================
        print(f"\n--- End-to-End Two-Stage Evaluation ---\n")

        e2e_results = end_to_end_evaluate(
            crop_model, phase2_model, test_df, args.data_dir,
            crop_to_diseases, img_size, args.batch_size
        )

        # Save E2E confusion matrix
        e2e_dir = os.path.join(seed_dir, "end_to_end")
        os.makedirs(e2e_dir, exist_ok=True)
        
        m = e2e_results["overall_metrics"]
        print(f"\nEnd-to-End Metrics (Seed {seed}):")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1_score']:.4f}\n")

        plot_confusion_matrix(
            e2e_results["y_true"], e2e_results["y_pred"],
            e2e_results["disease_names"],
            f"{args.model}_E2E_Hierarchical", seed, save_dir=e2e_dir
        )

        seed_results["end_to_end"] = {
            "overall_metrics": e2e_results["overall_metrics"],
            "error_propagation": e2e_results["error_propagation"],
            "total_infer_time_s": e2e_results["total_infer_time_s"],
            "disease_names": e2e_results["disease_names"]
        }

        # Print summary
        ep = e2e_results["error_propagation"]
        print(f"\n{'='*50}")
        print(f"RESULTS SUMMARY (Seed {seed})")
        print(f"{'='*50}")
        print(f"  Phase 1 Routing Accuracy:    {ep['routing_accuracy']:.4f}")
        print(f"  E2E Overall Accuracy:        {e2e_results['overall_metrics']['accuracy']:.4f}")
        print(f"  E2E F1 Score:                {e2e_results['overall_metrics']['f1_score']:.4f}")
        print(f"  Acc when routed correctly:    {ep['accuracy_when_routed_correctly']:.4f}")
        print(f"  Acc when routed incorrectly:  {ep['accuracy_when_routed_incorrectly']:.4f}")
        print(f"  Correctly routed:  {ep['correctly_routed']}/{ep['total_samples']}")
        print(f"  Incorrectly routed: {ep['incorrectly_routed']}/{ep['total_samples']}")
        print(f"{'='*50}\n")

        # Save all results for this seed
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(os.path.join(seed_dir, "hierarchical_results.json"), "w") as f:
            json.dump(seed_results, f, indent=4, default=convert)

        # Clear session
        tf.keras.backend.clear_session()

    print(f"\nAll hierarchical results saved to: {model_hier_dir}/")


if __name__ == "__main__":
    main()
