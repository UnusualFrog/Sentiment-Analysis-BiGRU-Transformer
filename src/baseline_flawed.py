"""
baseline_flawed.py
==================

This script replicates the work in Atlas et al. (2025) including several metholodgical errors:
  - SMOTE applied to the full dataset BEFORE the train/test split
  - No validation set (75/25 train/test only)
  - No fixed random seed (Omitted from the original paper)
 
The goal of this script is to demonstrate how inflated results can be achieved with poor methodology
"""
 
import re
import pandas as pd
import numpy as np
import torch
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

print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
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
 
df['Sentiment'] = df['Score'].apply(map_sentiment)
df = df.drop('Score', axis=1)

# Display pre-SMOTE class distribution
print("\nClass distribution before SMOTE:")
print(df['Sentiment'].value_counts())

# Split training features from target feature
X_text = df['Text']
y = df['Sentiment']

# # ==================== SMOTE Data Augmentation ====================
# Apply TF-IDF to convert text to numeric vectors which can be processed by SMOTE; NOTE: TF-IDF is not used for feature extraction
print("\nVectorizing text with TF-IDF for SMOTE input...")
# NOTE: Limited max_features value had to be used due to memory constraints
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X_text)

# Convert TF-IDF matrix to dense vector for SMOTE processing
X_dense = X_tfidf.toarray()
print(f"TF-IDF matrix shape: {X_dense.shape}")


# Apply SMOTE before split (generally incorrect, but accurate for replication)
print("\nApplying SMOTE to full dataset (before split)...")
smote = SMOTE(sampling_strategy='not majority')
X_resampled, y_resampled = smote.fit_resample(X_dense, y)
 
print(f"\nClass distribution after SMOTE:")
unique, counts = np.unique(y_resampled, return_counts=True)
for cls, cnt in zip(unique, counts):
    label = "";
    if cls == 0:
        label = "Negative"
    elif cls == 1:
        label = "Neutral"
    elif cls == 2:
        label = "Positive"
    print(f"  {label}: {cnt}")
 
print(f"\nTotal samples after SMOTE: {len(y_resampled)}")

# Reconstruct text data from feature matrix
print("\nReconstructing text sequences from resampled matrix...")

# Get shape of original data
n_original = len(X_text)
X_text_reset = X_text.reset_index(drop=True)
y_original = y.reset_index(drop=True).values
 
# for each class, build an array of original row indices for that class
class_indices = {
    cls: np.where(y_original == cls)[0]
    for cls in np.unique(y_original)
}
 
resampled_texts = []
for i in range(len(X_resampled)):
    # Original samples are placed at their original index
    if i < n_original:
        resampled_texts.append(X_text_reset.iloc[i])
    # Synethetic samples are inserted at a random position of the same class as an original sample
    else:
        # Synthetic sample — randomly draw a same-class original text.
        cls = y_resampled[i]
        rand_idx = np.random.choice(class_indices[cls])
        resampled_texts.append(X_text_reset.iloc[rand_idx])
 
X_text_resampled = pd.Series(resampled_texts).reset_index(drop=True)
print(f"Reconstructed text corpus size: {len(X_text_resampled)}")


# ==================== Text Pre-processing ====================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
 
# Apply full text preprocessing, including, lowercasing, removal of URLs, 
#   hashtags, @ mentions, punctuation & non alphabetic charcaters, 
#   word tokenization, stop word removal, single character and token removal
def preprocess(text: str) -> list:
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3. Remove hashtags and mentions
    text = re.sub(r'[@#]\S+', '', text)
    # 4. Remove punctuation / non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # 5. Tokenize
    tokens = word_tokenize(text)
    # 6. Lemmatize + 7. Stop word and single-char removal
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]
    return tokens
 
print("\nRunning NLP preprocessing pipeline...")
tokenized = X_text_resampled.apply(preprocess)
 
# Remove any reviews that became empty after preprocessing
mask = tokenized.apply(len) > 0
tokenized = tokenized[mask].reset_index(drop=True)
y_resampled = y_resampled[mask.values]
 
print(f"Samples after preprocessing: {len(tokenized)}")
print(f"\nSample preprocessed tokens (row 0):\n  {tokenized.iloc[0][:15]}")
