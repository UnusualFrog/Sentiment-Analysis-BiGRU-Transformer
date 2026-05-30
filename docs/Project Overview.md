# Project Methodology: Sentiment Analysis with BiGRU and Transformer Head

**Course:** CP4140 — Deep Learning & Neural Networks II  
**Based on:** Atlas et al. (2025), *A modernized approach to sentiment analysis of product reviews using BiGRU and RNN based LSTM deep learning models*, Scientific Reports  
**Target Journal:** Expert Systems with Applications (Elsevier) — IF ≈ 3.8  

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│         Phase 1 — Paper analysis & environment setup                │
│  Identify flaws · GitHub repo · pin seeds · folder structure        │
│  Label Amazon Fine Food Reviews (1–2★ neg · 3★ neu · 4–5★ pos)     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │    Two parallel baselines
              │                         │
              ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  Phase 2A — Flawed      │   │  Phase 2B — Corrected   │
│  baseline               │   │  baseline               │
│  (replicates paper)     │   │  (methodologically      │
│                         │   │   sound)                │
│ • 75/25 split           │   │ • 70/15/15 split        │
│ • SMOTE before split    │   │ • SMOTE after split     │
│ • No validation set     │   │ • Validation set used   │
│ • No fixed seed         │   │ • Fixed random seed     │
│ • Record inflated       │   │ • Record honest metrics │
│   metrics               │   │                         │
└────────────┬────────────┘   └────────────┬────────────┘
             │                             │
             │                             ▼
             │              ┌─────────────────────────────┐
             │              │  Phase 3 — Novel architecture│
             │              │  (built on corrected baseline)│
             │              │                              │
             │              │ • Replace LSTM with          │
             │              │   Transformer head           │
             │              │ • BiGRU → Self-Attention     │
             │              │   → Dense → Softmax          │
             │              │ • Same split, seed, preprocessing│
             │              │ • Log all hyperparameters    │
             └──────────────┴──────────────┬───────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Phase 4 — Evaluation & ablation                        │
│          All three models compared on the same corrected test set   │
│                                                                     │
│  • Comparison table: flawed baseline vs corrected baseline          │
│    vs BiGRU + Transformer                                           │
│  • Per-class F1, confusion matrix, ROC-AUC for each model           │
│  • Ablation: BiGRU-only · LSTM head · Transformer head              │
│  • Generate all figures: loss curves, ROC, confusion matrices       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│         Phase 5 — LaTeX paper (Expert Systems with Applications)    │
│                                                                     │
│  • Methodology critique section: explain inflated baseline numbers  │
│  • Results table spans all three models for direct comparison       │
│  • Ablation, discussion, 20+ references, 6,000+ words               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 6 — Code repository                         │
│   README · requirements.txt · separate scripts per model            │
│   saved weights · all figures                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Paper Analysis and Environment Setup

Before writing any code, the first phase establishes a full understanding of what is being replicated, what is being critiqued, and a clean, reproducible working environment.

### 1.1 Critical analysis of Atlas et al. (2025)

Read the paper carefully and document the following methodological problems, which will form the basis of the critique in the final paper:

- **SMOTE leakage:** SMOTE was almost certainly applied to the full dataset before the train/test split. This causes synthetic minority-class samples to appear in both the training and test sets, which inflates all reported metrics because the model is partially evaluated on data it has already seen in augmented form.
- **No validation set:** Only a train/test split is used. Without a validation set there is no principled way to monitor overfitting or tune hyperparameters, which further inflates reported test performance.
- **No random seed:** The paper reports no fixed seed, making results impossible to reproduce exactly and masking the variance that would reveal how unstable the metrics are.
- **Weak baselines:** The comparisons are against DCNN, MLP, CapsuleNet, and GAN. No transformer-based or BERT-based model is included, which is a significant omission given the state of the field in 2025.
- **Suspiciously perfect metrics:** The reported 98.79% accuracy and AUC values at or near 1.0 across five product categories are implausibly high for noisy, real-world Amazon reviews, and are consistent with the data leakage described above.

### 1.2 Dataset preparation

Download the **Amazon Fine Food Reviews** dataset from Kaggle (568,454 reviews). This is the closest publicly available Amazon product review corpus to the one used in the paper. Assign sentiment labels from star ratings as follows:

| Star rating | Sentiment label |
|-------------|-----------------|
| 1–2 stars   | Negative        |
| 3 stars     | Neutral         |
| 4–5 stars   | Positive        |

Inspect the class distribution before any splitting or augmentation and record it. The dataset is expected to be heavily skewed toward positive reviews.

### 1.3 Environment and repository setup

- Create a GitHub repository on day one and commit daily.
- Set a global random seed (e.g., `42`) in `numpy`, Python's `random` module, and the deep learning framework before any data operations.
- Create a `requirements.txt` immediately and pin all library versions.
- Recommended folder structure:

```
project/
├── data/               # raw and processed datasets
├── models/             # saved weights
├── results/            # CSV logs of all experimental runs
├── figures/            # PNG and PDF outputs
├── src/
│   ├── baseline_flawed.py
│   ├── baseline_corrected.py
│   ├── novel_model.py
│   ├── ablation.py
│   └── evaluate.py
├── paper/              # LaTeX source
├── README.md
└── requirements.txt
```

---

## Phase 2A — Flawed Baseline (Paper Replication)

The goal of this phase is to replicate the paper's experimental setup as faithfully as possible — including its methodological errors — in order to produce comparable inflated numbers that can be contrasted with the corrected results.

### 2A.1 Data setup

- Use a **75/25 train/test split** with no validation set, mirroring the paper's ratio.
- Apply **SMOTE to the full dataset before splitting**. This is the critical error being replicated. Fit SMOTE on all available data and then divide the SMOTE-augmented dataset into train and test portions.
- Do not fix a random seed (or use a different seed per run to illustrate variance), since the paper does not report one.

### 2A.2 Preprocessing

Apply the same NLP preprocessing pipeline described in the paper using NLTK:

1. Punctuation and whitespace removal
2. URL and hashtag removal
3. Named entity omission (reviewer names, brand names)
4. Tokenization (word-level)
5. Lemmatization
6. Stop word removal
7. Part-of-speech tagging

### 2A.3 Model architecture

Build the BiGRU + LSTM architecture as described in the paper:

- **Embedding layer:** Word2Vec embeddings (or GloVe as an alternative)
- **Feature extraction:** Bidirectional GRU layer (processes sequence in both forward and backward directions; concatenates outputs)
- **Classification:** LSTM layer with input, forget, and output gates
- **Output:** Dense layer with Softmax activation (3 classes: positive, negative, neutral)
- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy

### 2A.4 Training and recording

Train the model and record all of the following:

- Overall accuracy, precision, recall, macro F1-score
- Per-class precision, recall, and F1
- AUC (ROC)
- Training loss per epoch

These are the "inflated" numbers. They are expected to be high — potentially near the paper's reported 98.79% — and that is the intended outcome of this phase.

---

## Phase 2B — Corrected Baseline (Methodologically Sound)

This phase uses the identical BiGRU + LSTM architecture as Phase 2A, with only the experimental setup corrected. The purpose is to demonstrate what honest, reproducible results look like under sound methodology.

### 2B.1 Data setup

- Use a **70/15/15 train/validation/test split**.
- Fix the global random seed before any data operations.
- Apply **SMOTE to the training set only**, after splitting. Fit the SMOTE resampler on `X_train` and `y_train` exclusively. The validation and test sets must contain only original, non-augmented samples.

### 2B.2 Preprocessing

Apply the identical preprocessing pipeline as Phase 2A with the addition of duplicate rempvoal. The only differences are in the split, dupe removal and SMOTE application, not the text cleaning steps.

### 2B.3 Model architecture

Build the identical BiGRU + LSTM architecture as Phase 2A. No architectural changes are made between 2A and 2B — the only variables are the experimental setup decisions.

### 2B.4 Training and recording

Use the **validation set** to:

- Monitor training and validation loss per epoch to detect overfitting
- Tune hyperparameters (number of GRU units, dropout rate, learning rate)
- Determine the optimal number of training epochs (use early stopping if needed)

Evaluate the final trained model on the held-out **test set** and record the same metrics as Phase 2A:

- Overall accuracy, precision, recall, macro F1-score
- Per-class precision, recall, and F1
- AUC (ROC)

These are the honest baseline numbers. They will be lower than Phase 2A's, and the gap between them is your primary piece of evidence for the methodological critique. The test set established here is also the benchmark for all subsequent experiments — Phase 3 results must be evaluated on this same split.

---

## Phase 3 — Novel Architecture (BiGRU + Transformer Head)

This phase introduces the paper's novel contribution: replacing the LSTM classifier head with a self-attention Transformer head while keeping the BiGRU feature extractor unchanged.

### 3.1 Architecture

The full pipeline is:

```
Raw review text
       ↓
NLP Preprocessing (same as Phase 2B)
       ↓
Word2Vec Embeddings
       ↓
BiGRU Layer (bidirectional, returns full sequence of hidden states)
       ↓
Self-Attention Layer (attends over BiGRU hidden states)
       ↓
Global Pooling (weighted sum or mean of attended states)
       ↓
Dense Layer + Dropout
       ↓
Softmax (3 classes)
```

The self-attention mechanism allows the model to weight the importance of each position in the BiGRU output sequence, rather than relying on the final hidden state alone. This is a principled improvement because sentiment-relevant words may appear anywhere in a review, not just at the end.

### 3.2 Experimental controls

To ensure the comparison is controlled:

- Use the **identical 70/15/15 split and random seed** from Phase 2B
- Use the **identical preprocessing pipeline** from Phase 2B
- Evaluate on the **identical test set** from Phase 2B
- The only variable that changes relative to Phase 2B is the classifier head

Any performance gain over Phase 2B is therefore attributable solely to the architectural change.

### 3.3 Hyperparameter logging

Record every training run with:

- Learning rate, batch size, number of BiGRU units
- Number of attention heads (if multi-head)
- Dropout rate, number of training epochs
- Final train/validation/test metrics

This log is required for the reproducibility package and the ablation study.

---

## Phase 4 — Evaluation and Ablation

All three models — flawed baseline (2A), corrected baseline (2B), and BiGRU + Transformer (Phase 3) — are now compared in a structured evaluation.

### 4.1 Three-model comparison table

The primary results table in the paper will have the following structure:

| Model | Setup | Accuracy | Macro F1 | AUC |
|-------|-------|----------|----------|-----|
| BiGRU + LSTM (Phase 2A) | Flawed (SMOTE leak, no val set) | — | — | — |
| BiGRU + LSTM (Phase 2B) | Corrected | — | — | — |
| BiGRU + Transformer (Phase 3) | Corrected | — | — | — |

This table tells the complete story of the paper in a single view: the inflation caused by methodological errors, the honest baseline, and the genuine architectural improvement.

### 4.2 Per-class metrics

For each model, report precision, recall, and F1 broken down by class (positive, negative, neutral). Per-class metrics are especially important here because the Amazon dataset is likely heavily skewed toward positive reviews, meaning overall accuracy can be misleading. Improvements on the minority classes (negative, neutral) should be highlighted.

### 4.3 Ablation study

The ablation study isolates the contribution of each architectural component. All ablation runs use the corrected Phase 2B setup exclusively.

| Model variant | Accuracy | Macro F1 | Notes |
|---------------|----------|----------|-------|
| BiGRU only (dense output) | — | — | No sequential classifier |
| BiGRU + LSTM head | — | — | Phase 2B baseline |
| BiGRU + Transformer head | — | — | Phase 3 proposed model |

If dropout, attention pooling, or other components were added incrementally, add a row for each.

### 4.4 Figures to generate

Save all figures as both PNG and PDF:

- Training and validation loss curves per epoch (one per model)
- ROC curves for all models on the same axes
- Confusion matrices (one per model)
- Bar chart of per-class F1 for the ablation comparison
- Bar chart comparing overall accuracy across all three models

---

## Phase 5 — LaTeX Paper

**Target journal:** Expert Systems with Applications (Elsevier), IF ≈ 3.8  
**Backup journal:** Information Processing & Management (Elsevier), IF ≈ 4.0  
**Minimum length:** 6,000 words (excluding references and captions)  
**Minimum references:** 20, covering 2022–2026 where possible  

### 5.1 Paper structure

| Section | Key content |
|---------|-------------|
| Title, authors, abstract (≤250 words) | State the dual-baseline design and the BiGRU + Transformer contribution |
| Introduction | Problem motivation, research gap, three clearly bulleted contributions |
| Related Work (≥10 citations) | Original baseline paper + recent sentiment analysis + transformer NLP work |
| Methodology | Preprocessing pipeline, both baseline setups, novel architecture details, evaluation protocol |
| Experiments & Results | Three-model comparison table, per-class metrics, confusion matrices |
| Ablation Study | Component contribution table |
| Discussion | Interpret the inflation gap, interpret the architectural gain, limitations |
| Conclusion | Summary of contributions and future directions |
| References (≥20) | Elsevier numbered citation style |

### 5.2 Methodology critique subsection

The methodology section must include a dedicated subsection explaining why the flawed baseline (Phase 2A) produces unreliable numbers. Frame this not as an attack on the original authors but as a methodological contribution demonstrating the importance of correct SMOTE application and proper train/validation/test separation. Quantify the inflation: report the exact gap between 2A and 2B metrics as evidence.

### 5.3 Writing strategy

Write sections incrementally as each experiment is completed — do not leave all writing to the end:

- Write the methodology section while implementing the models
- Write results paragraphs immediately after each training run
- Write the discussion after the ablation study is complete
- Write the introduction and abstract last

---

## Phase 6 — Code Repository

The repository must allow a reader to independently reproduce every number in the paper.

### 6.1 Required files

| File / folder | Contents |
|---------------|----------|
| `README.md` | Step-by-step setup, dataset download instructions, command to run each experiment |
| `requirements.txt` | All dependencies with exact version pins |
| `src/baseline_flawed.py` | Phase 2A experiment (75/25, SMOTE before split) |
| `src/baseline_corrected.py` | Phase 2B experiment (70/15/15, SMOTE after split, fixed seed) |
| `src/novel_model.py` | Phase 3 BiGRU + Transformer training script |
| `src/ablation.py` | All ablation variants |
| `src/evaluate.py` | Shared evaluation utilities (metrics, figures) |
| `models/` | Saved weights (if under 100 MB) or Google Drive link |
| `figures/` | All paper figures as PNG and PDF |
| `results/` | CSV logs of all experimental runs with timestamps and hyperparameters |

### 6.2 README structure

The README should follow this structure:

1. Project overview and paper citation
2. Environment setup (conda/pip commands)
3. Dataset download and placement instructions
4. Commands to reproduce each experiment (one command per model)
5. Commands to regenerate all figures
6. Expected output metrics for each experiment (so readers can verify)

### 6.3 Commit discipline

Commit after every significant milestone: after each training run, after each figure is generated, after each paper section is drafted. The commit history serves as a timestamped record of original work.

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary dataset | Amazon Fine Food Reviews | Closest public equivalent to paper's private dataset; professor-recommended |
| Baseline 2A split | 75/25, no validation | Replicates paper exactly for direct comparison |
| Baseline 2B split | 70/15/15 | Methodologically correct; enables hyperparameter tuning |
| SMOTE in 2A | Before splitting | Intentional replication of paper's error |
| SMOTE in 2B | After splitting, training only | Prevents data leakage into validation/test sets |
| Novel architecture | BiGRU + Self-Attention head | Replaces LSTM; principled improvement; professor-specified |
| Ablation baseline | Phase 2B setup only | Keeps comparison controlled |
| Target journal | Expert Systems with Applications | IF ≈ 3.8; matches study plan |
