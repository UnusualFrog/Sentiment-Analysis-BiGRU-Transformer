# Sentiment Analysis: BiGRU + Transformer Head

**Course:** CP4140 — Deep Learning & Neural Networks II  
**Author:** Julia Valentine Forward  

A replication and methodological critique of Atlas et al. (2025), followed by a novel BiGRU + self-attention Transformer classifier head, all trained on the Amazon Product Reviews dataset.

---

## Project Overview

This project implements and compares three experimental conditions:

| Script | Model | Setup |
|---|---|---|
| `baseline_flawed.py` | BiGRU + LSTM | Replicates Atlas et al. (2025) including methodological errors (SMOTE before split, no validation set, no seed) |
| `baseline_corrected.py` | BiGRU + LSTM | Same architecture with all errors corrected (fixed seed, deduplication, 70/15/15 split, SMOTE after split) |
| `novel_model.py` | BiGRU + Transformer | Corrected setup with LSTM head replaced by multi-head self-attention + masked mean pooling |
| `ablation.py` | BiGRU-only, BiGRU+LSTM, BiGRU+Transformer | All three classifier heads compared under identical controlled conditions |
| `evaluate.py` | — | Generates all paper figures from saved prediction files |

---

## Repository Structure

```
project/
├── data/
│   └── Reviews.csv              ← place downloaded dataset here
├── models/                      ← saved model checkpoints (created at runtime)
├── results/
│   ├── logs/                    ← TensorBoard event files (created at runtime)
│   ├── baseline_flawed_results.csv
│   ├── baseline_corrected_results.csv
│   ├── novel_model_results.csv
│   ├── ablation_results.csv
│   ├── baseline_flawed_preds.npz
│   ├── baseline_corrected_preds.npz
│   ├── novel_model_preds.npz
│   └── ablation_*_preds.npz
├── figures/                     ← output figures (created at runtime)
├── src/
│   ├── baseline_flawed.py
│   ├── baseline_corrected.py
│   ├── novel_model.py
│   ├── ablation.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

> **Note:** The `models/`, `results/`, and `figures/` directories are created automatically when you run the scripts. You only need to create `data/` and place the dataset there manually.

---

## 1. Environment Setup

### Step 1 — Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Step 2 — Install PyTorch with CUDA 13.0 support

PyTorch must be installed **separately and first**, before all other packages, because it requires a custom index URL that cannot be expressed inside `requirements.txt`:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

### Step 3 — Verify GPU visibility before continuing

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output: `2.11.0 True`. If you see `False`, check your CUDA driver installation before proceeding.

### Step 4 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Download NLTK data

This is handled automatically at the start of each script, but you can pre-download it manually to avoid delays:

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
"
```

---

## 2. Dataset Download

The dataset is not included in this repository. It must be downloaded manually from Kaggle.

### Step 1 — Create a Kaggle account (if you do not have one)

Go to [https://www.kaggle.com](https://www.kaggle.com) and sign up for a free account.

### Step 2 — Navigate to the dataset page

Open this URL in your browser:

```
https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews
```

### Step 3 — Download the dataset

Click the **Download** button (top-right of the dataset page). Kaggle will package the dataset as a ZIP file named `amazon-product-reviews.zip` or similar.

### Step 4 — Extract and place the file

Extract the ZIP archive. Inside you will find `Reviews.csv`. Create the `data/` directory in the project root and move the file there:

```bash
mkdir -p data
mv /path/to/extracted/Reviews.csv data/Reviews.csv
```

## 3. Running the Experiments

All scripts are run from the **project root directory** (not from inside `src/`). Each script expects `data/Reviews.csv` to exist relative to where you run it.

> **Important:** The scripts must be run in order if you want to use `evaluate.py` to generate figures, because each training script saves `.npz` prediction files and `.csv` result files that `evaluate.py` reads. You can run them in any order otherwise.

### Flawed baseline (replicates Atlas et al. 2025 with data leakage)

```bash
python src/baseline_flawed.py
```

**Expected runtime:** ~25–40 minutes (10 epochs, no early stopping)  
**Outputs:**
- `results/baseline_flawed_results.csv` — accuracy, F1, AUC appended per run
- `results/baseline_flawed_preds.npz` — labels, predictions, class probabilities
- `results/logs/baseline_flawed_YYYYMMDD_HHMMSS/` — TensorBoard logs
- Terminal: per-epoch metrics, final classification report, confusion matrix

---

### Corrected baseline (methodologically sound replication)

```bash
python src/baseline_corrected.py
```

**Expected runtime:** ~15–25 minutes (early stopping typically triggers around epoch 4–6)  
**Outputs:**
- `results/baseline_corrected_results.csv`
- `results/baseline_corrected_preds.npz`
- `results/logs/baseline_corrected_YYYYMMDD_HHMMSS/`
- `models/baseline_corrected_YYYYMMDD_HHMMSS_best.pt` — best checkpoint by validation loss
- Terminal: per-epoch train/val metrics, early stopping trigger point, final test report

---

### Novel model (BiGRU + Transformer head)

```bash
python src/novel_model.py
```

**Expected runtime:** ~20–35 minutes (early stopping patience 5)  
**Outputs:**
- `results/novel_model_results.csv`
- `results/novel_model_preds.npz`
- `results/logs/novel_model_YYYYMMDD_HHMMSS/`
- `models/novel_model_YYYYMMDD_HHMMSS_best.pt`

---

### Ablation study (all three classifier heads in one run)

```bash
python src/ablation.py
```

**Expected runtime:** ~45–75 minutes (trains three full models sequentially)  
**Outputs:**
- `results/ablation_results.csv` — one row per variant
- `results/ablation_bigru_only_preds.npz`
- `results/ablation_bigru_lstm_preds.npz`
- `results/ablation_bigru_transformer_preds.npz`
- `figures/ablation_accuracy_bar.png`
- `figures/ablation_per_class_f1.png`
- `results/logs/ablation_*_YYYYMMDD_HHMMSS/` — one directory per variant
- Terminal: full ablation results table, per-class classification reports

---

### Generate evaluation figures

Run this **after** all three model scripts have completed at least one run each:

```bash
python src/evaluate.py
```

**Outputs (all saved to `figures/`):**
- `loss_curve_baseline_flawed.png`
- `loss_curve_baseline_corrected.png`
- `loss_curve_novel_model.png`
- `roc_curves_all_models.png`
- `confusion_matrix_baseline_flawed.png`
- `confusion_matrix_baseline_corrected.png`
- `confusion_matrix_novel_model.png`
- `accuracy_bar_all_models.png`
- Terminal: three-model comparison table, per-class metrics for all models

---

## 4. Monitoring Training with TensorBoard

Training metrics are logged in real time. To open the dashboard while a script is running (or after):

```bash
tensorboard --logdir=results/logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser. You will see loss curves, accuracy, F1, and AUC tracked per epoch for every run, with each run labeled by model name and timestamp.

---