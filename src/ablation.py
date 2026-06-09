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

