import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_single_phase_accuracy(results_dir='results', output_file='results/single_phase_accuracy_comparison.png'):
    model_accuracies = {}
    
    # List of baseline models to look for
    baseline_models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "ViT", "SwinTiny", "CvT", "Conformer"]
    
    for model_name in baseline_models:
        model_path = os.path.join(results_dir, model_name)
        if not os.path.exists(model_path):
            continue
            
        json_file = os.path.join(model_path, 'benchmark_results_seed42.json')
        if not os.path.exists(json_file):
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Structure: { ModelName: { 'seeds': { '42': { 'metrics': { 'accuracy': ... } } } } }
                accuracy = data[model_name]['seeds']['42']['metrics']['accuracy']
                
                # Give single-phase Conformer the main proposed name
                display_name = "MS-RoPE Conformer" if model_name == "Conformer" else model_name
                model_accuracies[display_name] = accuracy * 100 # Convert to percentage
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if not model_accuracies:
        print("No results found.")
        return

    # Sort models by accuracy for better visualization
    sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
    names, accs = zip(*sorted_models)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Custom vibrant palette
    colors = sns.color_palette("viridis", len(names))
    
    bars = plt.bar(names, accs, color=colors, edgecolor='black', alpha=0.8)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    plt.title('Single-Phase Global Baseline Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    
    # Also print a small table to console
    print("\n--- Single-Phase Results ---")
    print(f"{'Model':<20} | {'Accuracy':<10}")
    print("-" * 35)
    for name, acc in sorted_models:
        print(f"{name:<20} | {acc:.2f}%")

if __name__ == "__main__":
    plot_single_phase_accuracy()
