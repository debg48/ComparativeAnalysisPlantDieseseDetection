import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_parameters(results_dir='results', output_file='results/model_parameters_comparison.png'):
    model_params = {}
    
    # List of baseline models to look for (simplified comparison)
    baseline_models = ["ResNet50"]
    
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
                # Parameters are in the 'params' field (converted to Millions)
                params = data[model_name]['seeds']['42']['params']
                model_params[model_name] = params / 1e6
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Add Proposed Conformer from hierarchical results
    conformer_path = os.path.join(results_dir, 'Conformer_Hierarchical', 'seed42', 'hierarchical_results.json')
    if os.path.exists(conformer_path):
        try:
            with open(conformer_path, 'r') as f:
                data = json.load(f)
                # Sum phase 1 and phase 2 parameters
                p1_params = data['phase1_crop_router']['params']
                p2_params = data['phase2_specialist']['params']
                total_params = (p1_params + p2_params) / 1e6
                model_params['MS-RoPE Conformer'] = total_params
        except Exception as e:
            print(f"Error reading {conformer_path}: {e}")

    if not model_params:
        print("No results found.")
        return

    # Sort models by parameter count for better visualization
    sorted_models = sorted(model_params.items(), key=lambda x: x[1], reverse=True)
    names, params = zip(*sorted_models)

    # Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Create a divergent palette where 'Proposed' stands out if needed, or just use a nice one
    colors = sns.color_palette("rocket", len(names))
    
    # Optional: Highlight the proposed model with a specific color
    bar_colors = []
    for name in names:
        if name == 'MS-RoPE Conformer':
            bar_colors.append('#2ECC71') # Vibrant Green
        else:
            bar_colors.append('#3498DB') # Nice Blue

    bars = plt.bar(names, params, color=bar_colors, edgecolor='black', alpha=0.9)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(params) * 0.01), f"{yval:.2f}M", 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title('Model Parameter Count Comparison (In Millions)', fontsize=16, fontweight='bold', pad=25)
    plt.ylabel('Number of Parameters (Millions)', fontsize=13)
    plt.xlabel('Model Architecture', fontsize=13)
    
    # Log scale if ResNet50 is very dominant, but let's see. 23M vs 1.4M is ~16x. 
    # Let's keep linear but ensure ResNet doesn't crush others visually.
    plt.ylim(0, max(params) * 1.15)
    plt.xticks(rotation=20, ha='right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Parameter plot saved to: {output_file}")
    
    # Print table
    print("\n--- Parameter Counts ---")
    print(f"{'Model':<25} | {'Params (M)':<10}")
    print("-" * 40)
    for name, p in sorted_models:
        print(f"{name:<25} | {p:.2f}M")

if __name__ == "__main__":
    plot_model_parameters()
