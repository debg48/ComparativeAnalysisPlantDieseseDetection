import os
import json
import numpy as np
import argparse

def format_percent(val):
    return f"{val*100:.2f}%" if val is not None else "N/A"

def safe_mean(values):
    return np.mean(values) if values else None

def load_global_results(results_dir):
    models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT"]
    results = {}
    for m in models:
        m_dir = os.path.join(results_dir, m)
        if not os.path.exists(m_dir): continue
        for f in os.listdir(m_dir):
            if f.startswith("benchmark_results_seed") and f.endswith(".json"):
                seed = f.replace("benchmark_results_seed", "").replace(".json", "")
                with open(os.path.join(m_dir, f)) as fp:
                    data = json.load(fp)
                    if m not in results: results[m] = {}
                    if m in data and "seeds" in data[m]:
                        results[m][seed] = data[m]["seeds"][seed]
                    else:
                        results[m][seed] = data
    return results

def load_hierarchical_results(results_dir):
    models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT"]
    results = {}
    for m in models:
        h_dir = os.path.join(results_dir, f"{m}_Hierarchical")
        if not os.path.exists(h_dir): continue
        for seed_folder in os.listdir(h_dir):
            if not seed_folder.startswith("seed"): continue
            seed = seed_folder.replace("seed", "")
            json_path = os.path.join(h_dir, seed_folder, "hierarchical_results.json")
            if os.path.exists(json_path):
                with open(json_path) as fp:
                    if m not in results: results[m] = {}
                    results[m][seed] = json.load(fp)
    return results

def generate_report(results_dir, output_file):
    g_results = load_global_results(results_dir)
    h_results = load_hierarchical_results(results_dir)
    
    with open(output_file, "w") as f:
        f.write("# Crop Disease Detection: Experiments Report\n\n")
        
        # ==========================================
        # EXPERIMENT GROUP 1: Baseline Comparison
        # ==========================================
        f.write("## 🧪 EXPERIMENT GROUP 1: Baseline Comparison\n")
        f.write("**Purpose:** Show the proposed method is better (or meaningful) compared to global single-stage models.\n\n")
        
        f.write("| Model | Accuracy | F1 Score | Precision | Recall |\n")
        f.write("|---|---|---|---|---|\n")
        
        # Global models
        for m in ["ResNet50", "EfficientNetB0", "ViT", "CvT"]:
            if m in g_results:
                acc_list = [v["metrics"]["accuracy"] for v in g_results[m].values()]
                f1_list = [v["metrics"]["f1_score"] for v in g_results[m].values()]
                prec_list = [v["metrics"]["precision"] for v in g_results[m].values()]
                rec_list = [v["metrics"]["recall"] for v in g_results[m].values()]
                
                f.write(f"| {m} (Global) | {format_percent(safe_mean(acc_list))} | {format_percent(safe_mean(f1_list))} | {format_percent(safe_mean(prec_list))} | {format_percent(safe_mean(rec_list))} |\n")
        
        # Proposed Hierarchical (Two-Stage) - best model (assume CvT or the one available)
        best_h_model = "CvT" if "CvT" in h_results else list(h_results.keys())[0] if h_results else None
        if best_h_model:
            e2e_acc = [v["end_to_end"]["overall_metrics"]["accuracy"] for v in h_results[best_h_model].values()]
            e2e_f1 = [v["end_to_end"]["overall_metrics"]["f1_score"] for v in h_results[best_h_model].values()]
            e2e_prec = [v["end_to_end"]["overall_metrics"]["precision"] for v in h_results[best_h_model].values()]
            e2e_rec = [v["end_to_end"]["overall_metrics"]["recall"] for v in h_results[best_h_model].values()]
            f.write(f"| Two-Stage (Ours - {best_h_model}) | **{format_percent(safe_mean(e2e_acc))}** | **{format_percent(safe_mean(e2e_f1))}** | **{format_percent(safe_mean(e2e_prec))}** | **{format_percent(safe_mean(e2e_rec))}** |\n\n")
        else:
            f.write("| Two-Stage (Ours) | N/A | N/A | N/A | N/A |\n\n")


        # ==========================================
        # EXPERIMENT GROUP 2: Phase-wise Evaluation
        # ==========================================
        f.write("## 🧪 EXPERIMENT GROUP 2: Phase-wise Evaluation\n")
        f.write("**Purpose:** Understand each component's specialization benefit.\n\n")
        
        if best_h_model:
            # Phase 1
            p1_acc = [v["phase1_crop_router"]["metrics"]["accuracy"] for v in h_results[best_h_model].values()]
            f.write(f"### Phase 1 (Crop Classifier)\n")
            f.write(f"- **Accuracy:** {format_percent(safe_mean(p1_acc))}\n\n")
            
            # Phase 2
            f.write("### Phase 2 (Disease Classifiers per crop)\n")
            f.write("| Crop | Accuracy |\n")
            f.write("|---|---|\n")
            
            # Get all crops found in phase2
            crops = set()
            for seed_data in h_results[best_h_model].values():
                crops.update(seed_data.get("phase2_specialist", {}).get("per_crop_metrics", {}).keys())
                
            for crop in sorted(crops):
                crop_acc = []
                for v in h_results[best_h_model].values():
                    metrics = v.get("phase2_specialist", {}).get("per_crop_metrics", {}).get(crop)
                    if metrics:
                        crop_acc.append(metrics["accuracy"])
                f.write(f"| {crop} | {format_percent(safe_mean(crop_acc))} |\n")
            f.write("\n")
        else:
            f.write("*No Hierarchical model data available.*\n\n")

        # ==========================================
        # EXPERIMENT GROUP 3: Error Propagation
        # ==========================================
        f.write("## 🧪 EXPERIMENT GROUP 3: Error Propagation (🔥 CRITICAL)\n")
        f.write("**Purpose:** Deep insight into system weaknesses (impact of incorrect routing).\n\n")
        
        if best_h_model:
            acc_correct = [v["end_to_end"]["error_propagation"]["accuracy_when_routed_correctly"] for v in h_results[best_h_model].values()]
            acc_incorrect = [v["end_to_end"]["error_propagation"]["accuracy_when_routed_incorrectly"] for v in h_results[best_h_model].values()]
            
            f.write("| Condition | Final Accuracy |\n")
            f.write("|---|---|\n")
            f.write(f"| Phase 1 Correct (Routed correctly) | {format_percent(safe_mean(acc_correct))} |\n")
            f.write(f"| Phase 1 Wrong (Routed incorrectly) | {format_percent(safe_mean(acc_incorrect))} |\n\n")
        else:
            f.write("*No Hierarchical model data available.*\n\n")

        # ==========================================
        # EXPERIMENT GROUP 4: Per-Class Performance
        # ==========================================
        f.write("## 🧪 EXPERIMENT GROUP 4: Per-Class Performance\n")
        f.write("**Purpose:** Check imbalance + difficulty per disease class.\n\n")
        
        if best_h_model:
            f.write(f"### Per-Class F1 Score (Two-Stage {best_h_model})\n")
            
            # Read from the first available seed
            seed_key = list(h_results[best_h_model].keys())[0]
            e2e_data = h_results[best_h_model][seed_key]["end_to_end"]
            
            if "per_class_f1" in e2e_data["overall_metrics"] and "disease_names" in e2e_data:
                f1_scores = e2e_data["overall_metrics"]["per_class_f1"]
                class_names = e2e_data["disease_names"]
                
                f.write("| Class | F1 Score |\n")
                f.write("|---|---|\n")
                for c_name, score in zip(class_names, f1_scores):
                    f.write(f"| {c_name} | {format_percent(score)} |\n")
                f.write("\n")
                
                if "confusion_matrix" in e2e_data["overall_metrics"]:
                    cm = e2e_data["overall_metrics"]["confusion_matrix"]
                    f.write("*Raw confusion matrix was saved to results JSON. See plotted standard PNGs for heatmap visualizations.*\n\n")
            else:
                f.write("*Per-class F1 metric not found in JSON. Please re-run taking advantage of the updated tracking script.*\n\n")
        else:
            f.write("*No Hierarchical model data available.*\n\n")

        # ==========================================
        # EXPERIMENT GROUP 5: Computational Analysis
        # ==========================================
        f.write("## 🧪 EXPERIMENT GROUP 5: Computational Analysis\n")
        f.write("**Purpose:** Efficiency vs accuracy tradeoff.\n\n")
        f.write("| Model | Params | FLOPs | Train Time (s) | Infer Time (s) | Accuracy |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for m in ["ResNet50", "EfficientNetB0", "ViT", "CvT"]:
            if m in g_results:
                params_list = [v["params"] for v in g_results[m].values() if "params" in v]
                flops_list = [v["flops"] for v in g_results[m].values() if "flops" in v]
                tt_list = [v["train_time_s"] for v in g_results[m].values() if "train_time_s" in v]
                it_list = [v["infer_time_s"] for v in g_results[m].values() if "infer_time_s" in v]
                acc_list = [v["metrics"]["accuracy"] for v in g_results[m].values()]
                
                p = int(safe_mean(params_list)) if params_list else "N/A"
                fl = int(safe_mean(flops_list)) if flops_list else "N/A"
                tt = f"{safe_mean(tt_list):.1f}" if tt_list else "N/A"
                it = f"{safe_mean(it_list):.2f}" if it_list else "N/A"
                
                f.write(f"| {m} (Global) | {p} | {fl} | {tt} | {it} | {format_percent(safe_mean(acc_list))} |\n")
                
        if best_h_model:
            # For hierarchical, compute total params, inference time
            # Note: params = router_params + sum(specialist_params)
            # Train time = router_train + sum(specialist_train)
            params_list = []
            flops_list = []
            tt_list = []
            it_list = []
            acc_list = []
            
            for seed_data in h_results[best_h_model].values():
                router = seed_data["phase1_crop_router"]
                spec = seed_data["phase2_specialist"]
                
                tot_params = router["params"] + spec["params"]
                tot_flops = router["flops"] + spec["flops"]
                tot_tt = router["train_time_s"] + spec["train_time_s"]
                
                params_list.append(tot_params)
                flops_list.append(tot_flops)
                tt_list.append(tot_tt)
                it_list.append(seed_data["end_to_end"]["total_infer_time_s"])
                acc_list.append(seed_data["end_to_end"]["overall_metrics"]["accuracy"])
                
            p = int(safe_mean(params_list))
            fl = int(safe_mean(flops_list))
            tt = f"{safe_mean(tt_list):.1f}"
            it = f"{safe_mean(it_list):.2f}"
            
            f.write(f"| Two-Stage ({best_h_model}) | {p} | {fl} | {tt} | {it} | {format_percent(safe_mean(acc_list))} |\n")
            
    print(f"Report successfully saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Experiments Report Markdown")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--output", type=str, default="experiments_report.md", help="Output markdown file")
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)
