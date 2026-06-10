
"""
evaluate.py
==================
Generates visual figures for the report using matplotlib and model output data
 
Figures produced:
  1. Training loss curves per model
  2. ROC curves for all three models
  3. Confusion matrix per model
  4. Accuracy bar chart for all three models
 
Required inputs (run the three model scripts to generate these in /results/):
  baseline_flawed_results.csv
  baseline_corrected_results.csv
  novel_model_results.csv
  logs/baseline_flawed_{time_stamp}/        (TensorBoard files)
  logs/baseline_corrected_{time_stamp}/     (TensorBoard files)
  logs/novel_model_{time_stamp}/            (TensorBoard files)
  baseline_flawed_preds.npz      (labels, preds, probs)
  baseline_corrected_preds.npz   (labels, preds, probs)
  novel_model_preds.npz          (labels, preds, probs)
"""
 
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.preprocessing import label_binarize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 
 
# ==================== Config ====================
 
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
 
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = len(CLASS_NAMES)
 
# Consistent colour and label mapping used across all figures
MODEL_STYLES = {
    "baseline_flawed": {"label": "BiGRU+LSTM (Flawed)", "color": "#e74c3c", "linestyle": "--"},
    "baseline_corrected": {"label": "BiGRU+LSTM (Corrected)", "color": "#3498db", "linestyle": "-"},
    "novel_model": {"label": "BiGRU+Transformer", "color": "#2ecc71", "linestyle": "-"},
}
 
os.makedirs(FIGURES_DIR, exist_ok=True)
 
 
# ==================== Helper Functions ====================

# Save a matplotlib figure with a specified filename as a png file
def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)
 
# Load scalar data from the most recent tensorboard run of a specified model prefix
def load_tb_scalars(run_prefix, tag):
    # Construct a pattern for matching the directories for each tensorboard model run prefix (i.e., baseline_flawed, baseline_corrected, novel_model)
    pattern = os.path.join(LOGS_DIR, f"{run_prefix}_*")
    # Get all directories matching the pattern, sorted alphabetically (which is effectively sorted by time as only the timestamps differ under this pattern)
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        print(f"  ERROR: No TensorBoard run found for prefix '{run_prefix}'")
        return None, None
 
    # Use the most recently created run directory
    run_dir = candidates[-1]
    # Accumulator loads all tensorboard events from the run directory
    ea = EventAccumulator(run_dir)
    # Refresh available events returned by the accumulator
    ea.Reload()
    
    # Handle missing tags (ex. validation in baseline_flawed)
    if tag not in ea.Tags().get("scalars", []):
        print(f"  ERROR: Tag '{tag}' not found in {run_dir}")
        return None, None
    
    # Extract run data
    events = ea.Scalars(tag)
    steps = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# Load labels predictions and probabilities for the most recent run of each model
def load_preds(run_prefix):
    # Get path to most recent run's npz file, corresponding to the specified model (i.e., baseline_corrected)
    path = os.path.join(RESULTS_DIR, f"{run_prefix}_preds.npz")
    # Handle missing npz file
    if not os.path.exists(path):
        print(f"ERROR: Prediction file not found: {path}")
        print(f"    Run {run_prefix}.py first to generate it.")
        return None, None, None
    
    # Load and return npz array data (class labels, predictions and class probabilties)
    data  = np.load(path)
    return data["labels"], data["preds"], data["probs"]

# ==================== Figure 1: Loss Curves ====================
 
#  Plot loss curves for train/val/test sets, per epoch, with a seperate figure for each model
def plot_loss_curves():
    print("\n Plotting loss curves...")
 
    # flawed baseline uses 'Loss/test'; corrected and novel use 'Loss/val'
    val_tag_map = {
        "baseline_flawed": "Loss/test",
        "baseline_corrected": "Loss/val",
        "novel_model": "Loss/val",
    }
    val_label_map = {
        "baseline_flawed": "Test Loss",
        "baseline_corrected": "Val Loss",
        "novel_model": "Val Loss",
    }
    
    # loop through each of the three models
    for run_prefix, style in MODEL_STYLES.items():

        #  load tensorboard scalar values for both splits
        train_steps, train_vals = load_tb_scalars(run_prefix, "Loss/train")
        # Use test for flawed baseline, use val for others
        val_steps, val_vals = load_tb_scalars(run_prefix, val_tag_map[run_prefix])

        # Handle missing data
        if train_steps is None:
            print(f"  Skipping {run_prefix} - no TensorBoard data found.")
            continue
        
        # Generate blank figure
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot training loss
        ax.plot(train_steps, train_vals, color=style["color"],
                linestyle="--", linewidth=1.8, label="Train Loss")

        # plot validation loss
        if val_vals is not None:
            ax.plot(val_steps, val_vals, color=style["color"],
                    linestyle="-", linewidth=1.8, label=val_label_map[run_prefix])

        # Figure styling
        ax.set_title(f"Loss Curve - {style['label']}", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
 
        save_fig(fig, f"loss_curve_{run_prefix}")

# ==================== Figure 2: ROC Curves ====================
 
# plot one vs. rest ROC curve for all models
def plot_roc_curves():
    print("\n Plotting ROC curves...")
    
    # generate a blank figure with subplots for each class
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 4.5), sharey=True)
    
    # track if figure has been succesfully plotted
    any_plotted = False
    
    # Loop through each model
    for run_prefix, style in MODEL_STYLES.items():
        # Load prediction data from npz file
        labels, preds, probs = load_preds(run_prefix)

        # Handle missing data
        if labels is None:
            continue
        
        # only true if labels loaded correctly
        any_plotted = True

        # Binarise labels for one-vs-rest AUC computation
        labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

        # subplot for each class
        for cls_idx, cls_name in enumerate(CLASS_NAMES):\
            # produce false positve and true postive rates using roc_curve generation
            fpr, tpr, _ = roc_curve(labels_bin[:, cls_idx], probs[:, cls_idx])
            # calculate area under the ROC curve
            roc_auc = auc(fpr, tpr)

            # plot roc curve for current class
            axes[cls_idx].plot(
                fpr, tpr,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=f"{style['label']} (AUC={roc_auc:.3f})"
            )
    
    # Skip saving figure if nothing plotted
    if not any_plotted:
        print("  No prediction files found - skipping ROC figure.")
        plt.close(fig)
        return
    
    # Loop through and format subplots
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[cls_idx]
        # Add diagonal chance line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_title(f"ROC - {cls_name} (OvR)", fontsize=12)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        if cls_idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
 
    fig.suptitle("ROC Curves - All Models (One-vs-Rest)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "roc_curves_all_models")


# ==================== Figure 3: Confusion Matrices ====================
 
 # plot a confusion matrix for each model
def plot_confusion_matrices():
    print("\n Plotting confusion matrices...")
    
    # For each model
    for run_prefix, style in MODEL_STYLES.items():
        # Load prediction data from npz
        labels, preds, probs = load_preds(run_prefix)

        # Handle missing data
        if labels is None:
            continue
        
        # Generate a confusion matrix with normalization for better minority class visualization
        cm = confusion_matrix(labels, preds, normalize="true")

        # Generate a blank figure
        fig, ax = plt.subplots(figsize=(5, 4.5))
        # Generate a confusion matrix figure using the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        # Plot the confusion matrix to the figure
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")

        ax.set_title(f"Confusion Matrix - {style['label']}\n(row-normalised)", fontsize=12)
        fig.tight_layout()
        save_fig(fig, f"confusion_matrix_{run_prefix}")

# ==================== Figure 4: Overall Accuracy Bar Chart ====================
 
#  Plot overall accuracy for test set across all three models
def plot_accuracy_bar():
    print("\n Plotting accuracy bar chart...")
    
    # map model names to their csv outputs
    csv_map = {
        "baseline_flawed": os.path.join(RESULTS_DIR, "baseline_flawed_results.csv"),
        "baseline_corrected": os.path.join(RESULTS_DIR, "baseline_corrected_results.csv"),
        "novel_model": os.path.join(RESULTS_DIR, "novel_model_results.csv"),
    }
    
    # Initalize tracking variables
    labels_list = []
    accuracy_list = []
    color_list = []
    
    # Loop through all three models
    for run_prefix, style in MODEL_STYLES.items():
        # Get csv path
        csv_path = csv_map[run_prefix]

        # Handle missing file
        if not os.path.exists(csv_path):
            print(f"  ERROR CSV not found: {csv_path} - skipping.")
            continue

        # Read file data
        df  = pd.read_csv(csv_path)
        # Capture accuracy, labels and color for styling
        acc = df["accuracy"].iloc[-1]   # most recent run
        labels_list.append(style["label"])
        accuracy_list.append(acc)
        color_list.append(style["color"])
    
    # Handle missing data
    if not labels_list:
        print("  No CSV results found - skipping accuracy bar chart.")
        return
    
    # Generate blank figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Generate bar chart based on accuracy values
    bars = ax.bar(labels_list, accuracy_list, color=color_list, width=0.45, edgecolor="black", linewidth=0.7)
 
    # Annotate each bar with the exact accuracy value
    for bar, acc in zip(bars, accuracy_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    # Formatting & Styling
    ax.set_ylim(0, min(1.0, max(accuracy_list) + 0.08))
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Overall Test Accuracy - All Models", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "accuracy_bar_all_models")

# ==================== Print Three-Model Comparison Table ====================
 
# Generate a comparison table for easy comparison of all three models and their evaluation metrics
def print_comparison_table():
    # Map model names to output data
    csv_map = {
        "baseline_flawed": (os.path.join(RESULTS_DIR, "baseline_flawed_results.csv"),"Flawed (SMOTE leak, no val set)"),
        "baseline_corrected": (os.path.join(RESULTS_DIR, "baseline_corrected_results.csv"),"Corrected"),
        "novel_model": (os.path.join(RESULTS_DIR, "novel_model_results.csv"), "Corrected"),
    }
    
    # map model names to their detailed architecture
    model_labels = {
        "baseline_flawed": "BiGRU + LSTM",
        "baseline_corrected": "BiGRU + LSTM",
        "novel_model": "BiGRU + Transformer",
    }
 
    print("\n" + "=" * 75)
    print("THREE-MODEL COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Model':<30} {'Setup':<34} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 75)
    
    # Loop through each model
    for run_prefix, (csv_path, setup_label) in csv_map.items():
        # Handle missing data
        if not os.path.exists(csv_path):
            print(f"  {model_labels[run_prefix]:<30} CSV not found - run training script first.")
            continue
        # Read csv data
        df  = pd.read_csv(csv_path)
        # Get most recent row
        row = df.iloc[-1]
        # print formatted table
        print(
            f"{model_labels[run_prefix]:<30} "
            f"{setup_label:<34} "
            f"{row['accuracy']:.4f} "
            f"{row['f1']:.4f} "
            f"{row['auc']:.4f}"
        )
 
    print("=" * 75)

# ==================== Print Per-Class Metrics ====================
 
 # Print evaluation metrics for each class and each model
def print_per_class_metrics():
    print("\n" + "=" * 75)
    print("PER-CLASS METRICS")
    print("=" * 75)
    
    # Loop through each model
    for run_prefix, style in MODEL_STYLES.items():
        # Load npz data
        labels, preds, probs = load_preds(run_prefix)

        # Handle missing
        if labels is None:
            continue

        # Print formatted classification report for each model
        print(f"\n{style['label']}")
        print("-" * 50)
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))




# ==================== Main ====================
 
if __name__ == "__main__":
    print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
    print("---------- Model Evaluation: Figure Generation ----------\n")

    # prefix = "baseline_corrected"
    # print(load_tb_scalars(prefix, "Accuracy/train"))
    # print(load_preds(prefix))
    plot_loss_curves()
    plot_roc_curves()
    plot_confusion_matrices()
    plot_accuracy_bar()
    print_comparison_table()
    print_per_class_metrics()

    print("\n========= evaluate.py Complete - all figures saved to figures/ =========")
