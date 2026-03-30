import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_global_results(results_dir, model_name):
    """Load results from the flat global classifier."""
    model_dir = os.path.join(results_dir, model_name)
    results = {}
    if not os.path.exists(model_dir):
        return None
    for f in os.listdir(model_dir):
        if f.startswith("benchmark_results_seed") and f.endswith(".json"):
            seed = f.replace("benchmark_results_seed", "").replace(".json", "")
            with open(os.path.join(model_dir, f)) as fp:
                data = json.load(fp)
                results[seed] = data[model_name]["seeds"][int(seed)] if model_name in data else data
    return results if results else None


def load_hierarchical_results(results_dir, model_name):
    """Load results from the hierarchical classifier."""
    hier_dir = os.path.join(results_dir, f"{model_name}_Hierarchical")
    results = {}
    if not os.path.exists(hier_dir):
        return None
    for seed_folder in os.listdir(hier_dir):
        seed_path = os.path.join(hier_dir, seed_folder)
        if os.path.isdir(seed_path):
            json_path = os.path.join(seed_path, "hierarchical_results.json")
            if os.path.exists(json_path):
                with open(json_path) as fp:
                    results[seed_folder.replace("seed", "")] = json.load(fp)
    return results if results else None


def compare_model(model_name, results_dir, output_dir):
    """Generate comparison plots for a single model architecture."""
    global_results = load_global_results(results_dir, model_name)
    hier_results = load_hierarchical_results(results_dir, model_name)

    if global_results is None and hier_results is None:
        print(f"No results found for {model_name}")
        return

    model_output = os.path.join(output_dir, model_name)
    os.makedirs(model_output, exist_ok=True)

    # Collect metrics across seeds
    global_metrics = {"accuracy": [], "f1_score": [], "precision": [], "recall": []}
    hier_metrics = {"accuracy": [], "f1_score": [], "precision": [], "recall": []}
    hier_routing = {"routing_accuracy": [], "acc_correct_route": [], "acc_incorrect_route": []}
    hier_specialist = {}

    if global_results:
        for seed, data in global_results.items():
            for m in global_metrics:
                global_metrics[m].append(data["metrics"][m])

    if hier_results:
        for seed, data in hier_results.items():
            e2e = data.get("end_to_end", {})
            om = e2e.get("overall_metrics", {})
            ep = e2e.get("error_propagation", {})

            for m in hier_metrics:
                hier_metrics[m].append(om.get(m, 0))

            hier_routing["routing_accuracy"].append(ep.get("routing_accuracy", 0))
            hier_routing["acc_correct_route"].append(ep.get("accuracy_when_routed_correctly", 0))
            hier_routing["acc_incorrect_route"].append(ep.get("accuracy_when_routed_incorrectly", 0))

            # Per-crop specialist accuracy
            per_crop = data.get("phase2_specialist", {}).get("per_crop_metrics", {})
            for crop, metrics in per_crop.items():
                if metrics is None:
                    continue
                if crop not in hier_specialist:
                    hier_specialist[crop] = []
                hier_specialist[crop].append(metrics["accuracy"])

    # ============================================
    # PLOT 1: Global vs Hierarchical Bar Chart
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ["accuracy", "f1_score", "precision", "recall"]
    x = np.arange(len(metrics_names))
    width = 0.35

    global_means = [np.mean(global_metrics[m]) if global_metrics[m] else 0 for m in metrics_names]
    global_stds = [np.std(global_metrics[m]) if global_metrics[m] else 0 for m in metrics_names]
    hier_means = [np.mean(hier_metrics[m]) if hier_metrics[m] else 0 for m in metrics_names]
    hier_stds = [np.std(hier_metrics[m]) if hier_metrics[m] else 0 for m in metrics_names]

    bars1 = ax.bar(x - width/2, global_means, width, yerr=global_stds, label='Global (Flat)', color='#4A90D9', capsize=5)
    bars2 = ax.bar(x + width/2, hier_means, width, yerr=hier_stds, label='Hierarchical (Two-Stage)', color='#E8744F', capsize=5)

    ax.set_ylabel('Score')
    ax.set_title(f'{model_name}: Global vs Hierarchical Classification')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_names])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(model_output, "global_vs_hierarchical.png"), dpi=150)
    plt.close()

    # ============================================
    # PLOT 2: Error Propagation Analysis
    # ============================================
    if hier_routing["routing_accuracy"]:
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ["Phase 1\nRouting Acc", "Acc When\nCorrectly Routed", "Acc When\nIncorrectly Routed"]
        means = [
            np.mean(hier_routing["routing_accuracy"]),
            np.mean(hier_routing["acc_correct_route"]),
            np.mean(hier_routing["acc_incorrect_route"])
        ]
        colors = ['#2ECC71', '#3498DB', '#E74C3C']

        bars = ax.bar(categories, means, color=colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, means):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name}: Error Propagation Analysis')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output, "error_propagation.png"), dpi=150)
        plt.close()

    # ============================================
    # PLOT 3: Per-Crop Specialist Accuracy
    # ============================================
    if hier_specialist:
        fig, ax = plt.subplots(figsize=(10, 6))
        crop_names = sorted(hier_specialist.keys())
        crop_means = [np.mean(hier_specialist[c]) for c in crop_names]
        crop_stds = [np.std(hier_specialist[c]) for c in crop_names]

        colors = plt.cm.Set2(np.linspace(0, 1, len(crop_names)))
        bars = ax.bar(crop_names, crop_means, yerr=crop_stds, color=colors, capsize=5, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, crop_means):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name}: Per-Crop Specialist Accuracy')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output, "per_crop_specialist_accuracy.png"), dpi=150)
        plt.close()

    # ============================================
    # Save summary JSON
    # ============================================
    summary = {
        "model": model_name,
        "global": {
            "mean_accuracy": np.mean(global_metrics["accuracy"]) if global_metrics["accuracy"] else None,
            "mean_f1": np.mean(global_metrics["f1_score"]) if global_metrics["f1_score"] else None,
        },
        "hierarchical": {
            "mean_accuracy": np.mean(hier_metrics["accuracy"]) if hier_metrics["accuracy"] else None,
            "mean_f1": np.mean(hier_metrics["f1_score"]) if hier_metrics["f1_score"] else None,
            "mean_routing_accuracy": np.mean(hier_routing["routing_accuracy"]) if hier_routing["routing_accuracy"] else None,
        },
        "per_crop_specialist_accuracy": {c: np.mean(v) for c, v in hier_specialist.items()},
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(model_output, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=convert)

    print(f"Comparison charts saved to: {model_output}/")


def main():
    parser = argparse.ArgumentParser(description="Compare Global vs Hierarchical Results")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--model", type=str, default="all", help="Model to compare or 'all'")
    args = parser.parse_args()

    output_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    available_models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT"]

    if args.model == "all":
        models = available_models
    else:
        models = [args.model]

    for model_name in models:
        print(f"\n{'='*40}")
        print(f"Comparing: {model_name}")
        print(f"{'='*40}")
        compare_model(model_name, args.results_dir, output_dir)

    print(f"\nAll comparison results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
