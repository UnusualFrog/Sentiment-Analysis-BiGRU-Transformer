"""
ablation.py
==================
Runs the ablation study with no dependency imports (all models defined and trained in-script) to ensure maximum consistency
 
All three model variants are trained from scratch using the identical corrected setup:
  - Fixed random seed set
  - De-duplication of data
  - 70/15/15 train/val/test split
  - Same preprocessing pipeline, Word2Vec embeddings, and hyperparameters
  - SMOTE and word2vec embeddings applied to training set only

The only variable across runs is the classifier head attached to the BiGRU:
  model 1 - BiGRU only (global mean pool => Dense => Softmax)
  model 2 - BiGRU + LSTM head (identical to baseline_corrected.py)
  mdoel 3 - BiGRU + Transformer head (identical to novel_model.py)
"""
 
import re
import os
import gc
import random
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
 
# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
 
# Set Global Random Seed for reproducability
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
 
#  I/O directories
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models",    exist_ok=True)
 
# Sentiment-Class info
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = len(CLASS_NAMES)
 
print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
print("---------- ablation.py: Ablation Study ----------")
print("GPU active: ", torch.cuda.is_available())
print("\n")

# ==================== Data Exploration & Cleaning ====================
# Read raw dataset
df = pd.read_csv('data/Reviews.csv')
 
# Display dataset info and sample rows
df.info()
print(f"\n{df.head(3)}")
 
# Drop duplicate reviews (different users, same text content)
print(f"\nRows before de-duplication: {len(df)}")
df = df.drop_duplicates(subset='Text')
print(f"Rows after de-duplication: {len(df)}")
 
# Drop irrelevant features
df = df.drop(["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time", "Summary"], axis=1)
 
# Convert numeric rating ranges to three-class polarity
def map_sentiment(score):
    if score <= 2:
        return 0   # Negative
    elif score == 3:
        return 1   # Neutral
    else:
        return 2   # Positive

# Add numeric sentiment column
df['Sentiment'] = df['Score'].apply(map_sentiment)
# Drop text sentiment column
df = df.drop('Score', axis=1)
 
# Display pre-SMOTE class distribution
print("\nClass distribution before SMOTE:")
print(df['Sentiment'].value_counts())
 
# Split training features from target feature
X_text = df['Text']
y = df['Sentiment']
 
# Delete the dataframe and reclaim memory
del df
gc.collect()
 
 
# ==================== Train / Val / Test Split ====================
# 70/15/15 split performed BEFORE any augmentation or embedding training
# Fixed seed used for reproducibility
 
# First split off the 30% that will become val + test
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text, y,
    test_size=0.30,
    random_state=GLOBAL_SEED,
    stratify=y
)
 
# Split the remaining 30% evenly into val (15%) and test (15%)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp,
    test_size=0.50,
    random_state=GLOBAL_SEED,
    stratify=y_temp
)
 
# Free the full text series and temporary split from memory
del X_text, y, X_temp_text, y_temp
gc.collect()
 
# Reset indices on all splits for downstream consistency
X_train_text = X_train_text.reset_index(drop=True)
X_val_text   = X_val_text.reset_index(drop=True)
X_test_text  = X_test_text.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
 
#  Display split information
print(f"\nTrain size : {len(X_train_text)}")
print(f"Val size   : {len(X_val_text)}")
print(f"Test size  : {len(X_test_text)}")
 
# Loop through the train, val, and test sets and display the class distribution for each separately
for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"\n{split_name} class distribution:")
    for cls, label in zip([0, 1, 2], ['Negative', 'Neutral', 'Positive']):
        print(f"  {label}: {(split_y == cls).sum()}")
