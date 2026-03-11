import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"
LOG_FILE = EXP_DIR / "training.log"
CSV_FILE = EXP_DIR / "experiment_results.csv"

# Set style for professional look
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def parse_training_logs():
    """Parses training.log to extract loss and accuracy curves."""
    if not LOG_FILE.exists():
        print(f"Log file {LOG_FILE} not found.")
        return {}
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    
    history = {}
    current_model = None
    
    for line in lines:
        # Detect model start
        model_match = re.search(r"🚀 Training (\w+)...", line)
        if model_match:
            current_model = model_match.group(1)
            history[current_model] = {"loss": [], "val_loss": [], "acc": [], "f1": []}
            continue
        
        # Detect epoch stats
        stats_match = re.search(r"Epoch (\d+): Loss=([\d.]+), ValLoss=([\d.]+), Acc=([\d.]+), F1=([\d.]+)", line)
        if stats_match and current_model:
            history[current_model]["loss"].append(float(stats_match.group(2)))
            history[current_model]["val_loss"].append(float(stats_match.group(3)))
            history[current_model]["acc"].append(float(stats_match.group(4)))
            history[current_model]["f1"].append(float(stats_match.group(5)))
            
    return history

def plot_training_curves(history):
    """Plots training and validation loss curves."""
    if not history:
        return
    
    n_models = len(history)
    fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))
    
    if n_models == 1:
        axes = [axes]

    for i, (model_name, data) in enumerate(history.items()):
        # Loss plot
        axes[i][0].plot(data["loss"], label='Train Loss', marker='o', color='blue')
        axes[i][0].plot(data["val_loss"], label='Val Loss', marker='x', color='red')
        axes[i][0].set_title(f"{model_name} - Loss")
        axes[i][0].set_xlabel("Epoch")
        axes[i][0].set_ylabel("Loss")
        axes[i][0].legend()
        
        # Accuracy plot
        axes[i][1].plot(data["acc"], label='Val Accuracy', marker='o', color='green')
        axes[i][1].plot(data["f1"], label='Val F1', marker='x', color='orange')
        axes[i][1].set_title(f"{model_name} - Metrics")
        axes[i][1].set_xlabel("Epoch")
        axes[i][1].set_ylabel("Score")
        axes[i][1].legend()
        
    plt.tight_layout()
    save_path = EXP_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved training curves to {save_path}")

def plot_metrics_comparison(df):
    """Plots a bar chart comparing final metrics across models."""
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe for seaborn
    df_melted = df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], 
                        var_name="Metric", value_name="Value")
    
    ax = sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.ylim(0.0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=8)

    plt.tight_layout()
    save_path = EXP_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved metrics comparison to {save_path}")

def plot_confusion_matrices(df):
    """Plots confusion matrices for all models."""
    n_models = len(df)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
        
    for i, row in df.iterrows():
        cm = ast.literal_eval(row["Confusion_Matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{row['Model']}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_xticklabels(['Normal', 'Malicious'])
        axes[i].set_yticklabels(['Normal', 'Malicious'])
        
    plt.tight_layout()
    save_path = EXP_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrices to {save_path}")

def main():
    print("📊 Generating Visualizations...")
    
    # 1. Comparison Bar Chart and Confusion Matrices
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        plot_metrics_comparison(df)
        plot_confusion_matrices(df)
    else:
        print(f"CSV file {CSV_FILE} not found.")
        
    # 2. Training Curves
    history = parse_training_logs()
    if history:
        plot_training_curves(history)
    
    print("✨ All graphs generated successfully in the 'experiments/' directory.")

if __name__ == "__main__":
    main()
