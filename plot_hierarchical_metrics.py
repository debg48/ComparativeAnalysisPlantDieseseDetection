import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_hierarchical_metrics(model_name="Conformer_Hierarchical", seed="seed42", results_dir="results"):
    file_path = os.path.join(results_dir, model_name, seed, "hierarchical_results.json")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    phases = [
        ("Phase 1 (Router)", data.get("phase1_crop_router", {}).get("metrics", {})),
        ("Phase 2 (Specialist)", data.get("phase2_specialist", {}).get("metrics", {})),
        ("End-to-End", data.get("end_to_end", {}).get("overall_metrics", {}))
    ]
    
    # Extract data
    bar_data = {metric: [] for metric in metrics_keys}
    phase_names = []
    
    for phase_name, phase_metrics in phases:
        phase_names.append(phase_name)
        for metric in metrics_keys:
            val = phase_metrics.get(metric, 0) * 100
            bar_data[metric].append(val)
            
    # Plotting
    x = np.arange(len(phase_names))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Use a visually distinct palette
    colors = sns.color_palette("muted", len(metrics_keys))
    
    for i, metric in enumerate(metrics_keys):
        offset = (i - 1.5) * width
        bars = plt.bar(x + offset, bar_data[metric], width, label=display_names[i], color=colors[i], edgecolor='black', alpha=0.9)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.title(f'Hierarchical Performance: {model_name.replace("_", " ")}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Pipeline Stage', fontsize=12, fontweight='bold')
    plt.xticks(x, phase_names, fontsize=12)
    
    # Adjust y-limits and legend
    plt.ylim(0, 115) 
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(loc='upper right', ncol=4, fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(results_dir, f"{model_name}_hierarchical_metrics.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    
if __name__ == "__main__":
    plot_hierarchical_metrics()
    plot_hierarchical_metrics("CvT_Hierarchical")
