"""
baseline_corrected.py
==================

This script replicates the methodology in Atlas et al. (2025) while correcting several metholodgical errors:
  - SMOTE applied After train/test split
  - Validation set used (70/15/15 train/val/test)
  - Random seed applied
 
The goal of this script is to demonstrate the actual results of the replicated methodlogy when correctly applied
"""
import random
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

# Download required NLTK resources (safe to re-run; skips if already present)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set Global Random Seed for reproducability
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
# tf.random.set_seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
print("---------- Corrected Baseline ----------")
print("GPU active: ", torch.cuda.is_available())
print("\n")

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
 
df['Sentiment'] = df['Score'].apply(map_sentiment)
df = df.drop('Score', axis=1)

# Display pre-SMOTE class distribution
print("\nClass distribution before SMOTE:")
print(df['Sentiment'].value_counts())

# Split training features from target feature
X_text = df['Text']
y = df['Sentiment']
