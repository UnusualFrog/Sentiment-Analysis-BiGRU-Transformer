
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
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
 
CLASS_NAMES  = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES  = len(CLASS_NAMES)
 
# Consistent colour and label mapping used across all figures
MODEL_STYLES = {
    "baseline_flawed":     {"label": "BiGRU+LSTM (Flawed)",      "color": "#e74c3c", "linestyle": "--"},
    "baseline_corrected":  {"label": "BiGRU+LSTM (Corrected)",   "color": "#3498db", "linestyle": "-"},
    "novel_model":         {"label": "BiGRU+Transformer",         "color": "#2ecc71", "linestyle": "-"},
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
        print(f"  [WARN] No TensorBoard run found for prefix '{run_prefix}'")
        return None, None
 
    # Use the most recently created run directory
    run_dir = candidates[-1]
    # Accumulator loads all tensorboard events from the run directory
    ea = EventAccumulator(run_dir)
    # Refresh available events returned by the accumulator
    ea.Reload()
    
    # Handle missing tags (ex. validation in baseline_flawed)
    if tag not in ea.Tags().get("scalars", []):
        print(f"  [WARN] Tag '{tag}' not found in {run_dir}")
        return None, None
    
    # Extract run data
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
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

# ==================== Main ====================
 
if __name__ == "__main__":
    print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
    print("---------- Model Evaluation: Figure Generation ----------\n")

    prefix = "baseline_corrected"
    print(load_tb_scalars(prefix, "Accuracy/train"))
    print(load_preds(prefix))

    print("\n========= evaluate.py Complete — all figures saved to figures/ =========")
