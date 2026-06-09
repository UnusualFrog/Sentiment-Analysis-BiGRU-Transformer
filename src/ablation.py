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

# ==================== SMOTE Data Augmentation ====================
# Apply TF-IDF to convert text to numeric vectors which can be processed by SMOTE; NOTE: TF-IDF is not used for feature extraction
print("\nVectorizing training text with TF-IDF for SMOTE input...")
# NOTE: max features=500 and float32 were chosen to adhere to memory constraints of hardware used
vectorizer = TfidfVectorizer(max_features=500, dtype=np.float32)
 
# Fit and transform only the training set to prevent leakage into val/test
X_tfidf_train = vectorizer.fit_transform(X_train_text)
 
# Convert TF-IDF matrix to dense vector for SMOTE processing
X_dense_train = X_tfidf_train.toarray()
 
# Free the sparse matrix to save memory
del X_tfidf_train
gc.collect()
 
print(f"TF-IDF matrix shape (train): {X_dense_train.shape}")
 
# Apply SMOTE to the training set only (correct methodology - no leakage into val/test)
print("\nApplying SMOTE to training set only (after split)...")
smote = SMOTE(sampling_strategy='not majority', random_state=GLOBAL_SEED)
X_resampled, y_resampled = smote.fit_resample(X_dense_train, y_train)
 
# Free X_dense_train for memory now that SMOTE is complete
del smote, X_dense_train
gc.collect()
 
print(f"\nClass distribution after SMOTE (train only):")
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
 
print(f"\nTotal training samples after SMOTE: {len(y_resampled)}")
 
# Reconstruct text data from feature matrix
print("\nReconstructing text sequences from resampled matrix...")
 
# Get shape of original training data
n_original = len(X_train_text)
y_train_values = y_train.values
 
# Free the original y_train series - y_train_values is the working copy
del y_train
gc.collect()
 
# for each class, build an array of original row indices for that class
class_indices = {
    cls: np.where(y_train_values == cls)[0]
    for cls in np.unique(y_train_values)
}
 
resampled_texts = []
for i in range(len(X_resampled)):
    # Original samples are placed at their original index
    if i < n_original:
        resampled_texts.append(X_train_text.iloc[i])
    # Synethetic samples are inserted at a random position of the same class as an original sample
    else:
        cls = y_resampled[i]
        rand_idx = np.random.choice(class_indices[cls])
        resampled_texts.append(X_train_text.iloc[rand_idx])
 
# Free the numerical resampled matrix for memory
del X_resampled, X_train_text, class_indices, y_train_values
gc.collect()
 
#  Convert to array to data series
X_train_text_resampled = pd.Series(resampled_texts).reset_index(drop=True)
# delete array for memory
del resampled_texts
gc.collect()
 
print(f"Reconstructed training text corpus size: {len(X_train_text_resampled)}")
 
 
# ==================== Text Pre-processing ====================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
 
# Apply full text preprocessing, including, lowercasing, removal of URLs,
#   hashtags, @ mentions, punctuation & non alphabetic charcaters,
#   word tokenization, stop word removal, single character and token removal
def preprocess(text):
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
 
print("\nRunning NLP preprocessing pipeline on all splits...")
tokenized_train = X_train_text_resampled.apply(preprocess)
tokenized_val   = X_val_text.apply(preprocess)
tokenized_test  = X_test_text.apply(preprocess)
 
# Free the raw text series for memory
del X_train_text_resampled, X_val_text, X_test_text
gc.collect()
 
# Remove any reviews that became empty after preprocessing
mask_train = tokenized_train.apply(len) > 0
tokenized_train = tokenized_train[mask_train].reset_index(drop=True)
y_resampled     = y_resampled[mask_train.values]
 
mask_val = tokenized_val.apply(len) > 0
tokenized_val = tokenized_val[mask_val].reset_index(drop=True)
y_val = y_val[mask_val.values].reset_index(drop=True)
 
mask_test = tokenized_test.apply(len) > 0
tokenized_test = tokenized_test[mask_test].reset_index(drop=True)
y_test = y_test[mask_test.values].reset_index(drop=True)
 
print(f"Samples after preprocessing - Train: {len(tokenized_train)}, Val: {len(tokenized_val)}, Test: {len(tokenized_test)}")
print(f"\nSample preprocessed tokens (train row 0):\n  {tokenized_train.iloc[0][:15]}")

# ==================== Word2Vec Embeddings ====================
# Word2Vec is trained on the training corpus only to prevent leakage into val/test
 
print("\nTraining Word2Vec on training corpus only (after split)...")
# 100d vector for every word
W2V_DIM = 100
# 5 words of surrounding context per word
W2V_WINDOW = 5
# Words appearing less than twice are considered noise and are filtered out
W2V_MINCOUNT = 2
 
# Construct embedding vocabulary linking semantically related words
#   for downstream consumption by BiGRU
w2v_model = Word2Vec(
    sentences = tokenized_train.tolist(),
    vector_size = W2V_DIM,
    window = W2V_WINDOW,
    min_count = W2V_MINCOUNT,
    workers = 4,
    sg = 0,
    seed = GLOBAL_SEED,
)
 
vocab_size = len(w2v_model.wv)
print(f"Word2Vec vocabulary size: {vocab_size}")
 
 
# ==================== Vocabulary Index & Embedding Matrix ====================
# Reserve index 0 and 1 for padding and UNK tokens respectively
PAD_IDX = 0
UNK_IDX = 1
 
# Map each word in vocab to unique integer for downstream BiGRU consumption
word2idx = {word: idx + 2 for idx, word in enumerate(w2v_model.wv.index_to_key)}
 
# Generate a lookup matrix of size equal to the vocabulary + 2 for reserved tokens
embedding_matrix = np.zeros((vocab_size + 2, W2V_DIM), dtype=np.float32)
# Map each word's matrix to a row in the embedding matrix
#   allowing lookup of a word's matrix by index
for word, idx in word2idx.items():
    embedding_matrix[idx] = w2v_model.wv[word]
 
# Free the word2vec model from memory
del w2v_model
gc.collect()
 
print(f"Embedding matrix shape: {embedding_matrix.shape}")
 
 
# ==================== Encode & Pad Sequences ====================
 
# Generate a list of indexes corresponding to each word in the vocabulary
# words not found in the vocabulary are replaced with the index 1, respresenting the UNK token
def encode(tokens):
    return [word2idx.get(tok, UNK_IDX) for tok in tokens]
 
encoded_train = tokenized_train.apply(encode)
encoded_val   = tokenized_val.apply(encode)
encoded_test  = tokenized_test.apply(encode)
 
# Free tokenized series from memory
del tokenized_train, tokenized_val, tokenized_test
gc.collect()
 
# Get length of each sequence (computed from training set only to prevent leakage)
lengths = encoded_train.apply(len)
 
# Set max length as 95% of the longest sequence (derived from training set only)
MAX_LEN = int(np.percentile(lengths, 95))
 
# Show range of sequence lengths
print(f"\nSequence length - min: {lengths.min()}, "
      f"mean: {lengths.mean():.0f}, 95th pct: {MAX_LEN}, max: {lengths.max()}")
 
# Pads or truncates a sequence until it reaches max length
def pad_or_truncate(seq, max_len):
    # truncate long sequences
    seq = seq[:max_len]
 
    # pad short sequences to max length
    return seq + [PAD_IDX] * (max_len - len(seq))
 
# Apply pad_or_truncate to all sequences
X_train_padded = np.array(
    [pad_or_truncate(seq, MAX_LEN) for seq in encoded_train],
    dtype=np.int64
)
X_val_padded = np.array(
    [pad_or_truncate(seq, MAX_LEN) for seq in encoded_val],
    dtype=np.int64
)
X_test_padded = np.array(
    [pad_or_truncate(seq, MAX_LEN) for seq in encoded_test],
    dtype=np.int64
)
 
# Convert the targets into array format
y_train_array = np.array(y_resampled, dtype=np.int64)
y_val_array = np.array(y_val, dtype=np.int64)
y_test_array = np.array(y_test, dtype=np.int64)
 
# Free encoded sequences and label series from memory
del encoded_train, encoded_val, encoded_test, y_resampled, y_val, y_test
gc.collect()
 
print(f"\nFinal padded input shapes - Train: {X_train_padded.shape}, Val: {X_val_padded.shape}, Test: {X_test_padded.shape}")
print(f"Final label array shapes  - Train: {y_train_array.shape}, Val: {y_val_array.shape}, Test: {y_test_array.shape}")
