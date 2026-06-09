
"""
novel_model.py
==================
This script introduces the novel architectural contribution of this project:
  - LSTM classifier head replaced with a self-attention Transformer head
  - All other arcitechture left as-is
Any performance gain over baseline_corrected.py is attributable solely
to the architectural change from LSTM to self-attention.
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
import torch.nn.functional as F
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

print("========= CP4140 - Sentiment Analysis BiGRU Transformer =========")
print("---------- Novel Model: BiGRU + Transformer Head ----------")
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

# Apply SMOTE to the training set only (correct methodology, no leakage into val/test)
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

# Free the original y_train series, y_train_values is the working copy
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

X_train_text_resampled = pd.Series(resampled_texts).reset_index(drop=True)
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

# Get length of each sequence - computed from training set only to prevent leakage
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
y_val_array   = np.array(y_val,       dtype=np.int64)
y_test_array  = np.array(y_test,      dtype=np.int64)

# Free encoded sequences and label series from memory
del encoded_train, encoded_val, encoded_test, y_resampled, y_val, y_test
gc.collect()

print(f"\nFinal padded input shapes - Train: {X_train_padded.shape}, Val: {X_val_padded.shape}, Test: {X_test_padded.shape}")
print(f"Final label array shapes  - Train: {y_train_array.shape}, Val: {y_val_array.shape}, Test: {y_test_array.shape}")


# ==================== PyTorch Dataset & DataLoader ====================

# Wrapper class for exposing tensors attributes through a streamlined interface
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    # Get count of total samples
    def __len__(self) -> int:
        return len(self.y)

    # Get the sequence corresponding to the provided word index
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


BATCH_SIZE = 64

train_loader = DataLoader(ReviewDataset(X_train_padded, y_train_array), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(ReviewDataset(X_val_padded,   y_val_array),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(ReviewDataset(X_test_padded,  y_test_array),  batch_size=BATCH_SIZE, shuffle=False)

# Free numpy splits from memory as dataloader holds copies
del X_train_padded, X_val_padded, X_test_padded, y_train_array, y_val_array, y_test_array
gc.collect()

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Test batches  : {len(test_loader)}")
print("\n========= Preprocessing Complete =========")

# ==================== Model Definition ====================

# Self attention over the BiGRU output seqeunce to compute the weights of token dependencies between tokens in the sequence
# IMPROVEMENT: this is an imporvement over the original work's LTSM head which used the final hidden state
#               which can reduce the signal of early tokens in a sequence. Self attention solves this by
#               calculating attention weights for each token with all other tokens pair-wise
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        # Inherit properties from the base PyTorch neural network class
        super(SelfAttention, self).__init__()

        # Multihead splits BiGRU input to learn different token dependency relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
 
    def forward(self, x, key_padding_mask=None):
        # Query, key, and value are all the same BiGRU output sequence (self-attention)
        attn_out, attn_weights = self.attention(
            query=x, # each word looks for other words it should focus on
            key=x, # words used to measure how related they are
            value=x, # information gathered from relatedness to other words
            key_padding_mask=key_padding_mask
        )

        return attn_out, attn_weights
    
# BiGRU + Transformer Head architecture
# IMPROVEMENT: LTSM classifierreplaced with self-attention+mean pooling over the attended sequence
#               This allows the model to better capture long-range dependencies as LTSM is biased
#               towards later entries in a sequence due to use of the final hidden state. Instead
#               the self attention of the Transformer head allows for token dependency to be 
#               measured at any distance
class BiGRUTransformer(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_heads, num_classes, dropout):
        # Inherit properties from the base PyTorch neural network class
        super(BiGRUTransformer, self).__init__()
 
        # Get vocab size and dimensions
        vocab_size, embed_dim = embedding_matrix.shape
 
        # Embedding layer initialised with pre-trained Word2Vec vocabulary weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
 
        # Unfreeze the embedding layer to allow for fine-tuning during training
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=True)
 
        # BiGRU layer processes the embedded sequence in both forward and backwards directions to capture broad word context
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
 
        # BiGRU output dimension is hidden_dim * 2 (forward + backward concatenated);
        # this feeds directly into the self-attention layer in place of the LSTM
        gru_out_dim = hidden_dim * 2
 
        # Self-attention layer attends over all BiGRU hidden states simultaneously
        self.attention = SelfAttention(input_dim=gru_out_dim, num_heads=num_heads)
 
        # Apply dropout layer before classification for regularization
        self.dropout = nn.Dropout(dropout)
 
        # Dense output layer maps the mean-pooled attended states to class logits
        # (softmax is auto-applied by CEL)
        self.fc = nn.Linear(gru_out_dim, num_classes)
 
    def forward(self, x):
        # BiGRU batch_first means x = (batch, seq_len)
 
        # Build a boolean padding mask: True where the token is PAD_IDX which are assigned zero weight
        padding_mask = (x == PAD_IDX)
 
        # dropout applied after embeddings layer to prevent overfitting on vocabulary
        embedded = self.dropout(self.embedding(x))
 
        # apply BiGRU to embedding output
        # IMPROVEMENT: full sequence of hidden states is retained (not just the final state) so attention can operate over every position
        gru_out, _ = self.bigru(embedded)
 
        # apply dropout to BiGRU output
        gru_out = self.dropout(gru_out)
 
        # apply self-attention over the full BiGRU sequence
        attn_out, _ = self.attention(gru_out, key_padding_mask=padding_mask)
 
        # global mean pooling over the attended sequence collapses (batch, seq_len, dim)
        # to (batch, dim); mean is taken only over non-padding positions to avoid
        # diluting the representation with zero-padded slots

        # convert padding tokens to value 0
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # get count of non-padding tokens, inverse padding mask grabs any non-padding tokens 
        # NOTE: clamped to min of 1 to prevent zero-division for seqeunces of all padding
        non_pad_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        # mean pooling of attention seqeunce (padding removed)
        pooled = attn_out.sum(dim=1) / non_pad_counts
 
        # apply dropout before classification
        out = self.dropout(pooled)
        # compute raw logits (softmax applied by CEL)
        logits = self.fc(out)
 
        return logits

# ==================== Model Initialisation ====================
 
# Hyperparameters -kept identical to baseline_corrected (excluding transformer params) for proper comparison
HIDDEN_DIM = 128   # GRU units per direction (128 forwards, 128 backwards = 256 total)
NUM_HEADS = 4     # attention heads; gru_out_dim (256) must be divisible by NUM_HEADS for clean sharing of sequences
NUM_CLASSES = 3     # 3-class sentiment classification
DROPOUT = 0.3   # 30% dropout rate
EPOCHS = 10    # 10 Training epochs
LR = 1e-3  # common baseline learning rate
PATIENCE = 5     # early stopping patience
 
# Verify GPU available before training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
 
# Initialize model with hyperparameters on the GPU
model = BiGRUTransformer(
    embedding_matrix=embedding_matrix,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)
 
# Free the embedding matrix from memory
del embedding_matrix
gc.collect()
 
print(model)
print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
# Cross Entropy used for Loss (Softmax applied internally)
criterion = nn.CrossEntropyLoss()
 
# Adam optimiser (assumed based on common practicces and reference in original work's literature review)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
print("\n========= Model Initialised - Ready for Training =========")

# ==================== TensorBoard Writer ====================
 
# Each run is logged to a timestamped subdirectory so runs dont overwrite each other
run_name = f"novel_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Log each run to the results directory with filename logs_{current_run_name}
writer = SummaryWriter(log_dir=os.path.join("results", "logs", run_name))
 
# ==================== Training Loop ====================

def train_epoch(model, loader, criterion, optimizer, device):
    # Set model to training model and initalize tracking variables
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
 
    # Loop through each seqeunce in the data loader
    for sequences, labels in loader:
        # Use GPU
        sequences = sequences.to(device)
        labels = labels.to(device)
 
        # reset gradients
        optimizer.zero_grad()
        # compute forward pass to produce raw logits
        logits = model(sequences)
        # pass logits to CEL for softmax classification and loss calculation
        loss = criterion(logits, labels)
        # compute backwards pass
        loss.backward()
        # update weights using LR step magnitude
        optimizer.step()
 
        # track loss and predictions to calculate evaluation metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
 
    # compute average loss and accuracy
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

# ==================== Evaluation ====================
 
def evaluate(model, loader, criterion, device):
    # Set model to eval mode to disable dropout during inference
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
 
    # Disable gradient computation during evaluation (no updates)
    with torch.no_grad():
        # Loop through each sequence
        for sequences, labels in loader:
            # Use GPU
            sequences = sequences.to(device)
            labels = labels.to(device)
 
            # forward pass
            logits = model(sequences)
            # softmax & loss
            loss = criterion(logits, labels)
            # track loss
            total_loss += loss.item()
 
            # Convert logits to probabilities for AUC computation
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
 
            # track outputs for evaluation metric computations
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    avg_loss = total_loss / len(loader)
    all_probs = np.array(all_probs)
 
    # Compute evaluation metrics
    metrics = {
        "loss":      avg_loss,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1":        f1_score(all_labels, all_preds, average='macro', zero_division=0),
        "auc":       roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'),
    }
 
    return metrics, all_preds, all_labels, all_probs

# ==================== Run Training ====================
 
print(f"\nStarting training - {EPOCHS} total epochs (early stopping patience: {PATIENCE})")
print(f"TensorBoard run: results/logs/{run_name}\n")
 
# Early stopping state
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_path = os.path.join("models", f"{run_name}_best.pt")
os.makedirs("models", exist_ok=True)
 
#  Loop through epochs
for epoch in range(1, EPOCHS + 1):
    # train model
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    # evaluate on validation set to monitor for overfitting
    val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
 
    # Write evaluation metrics to log file for tensorboard tracking
    writer.add_scalar("Loss/train",     train_loss,              epoch)
    writer.add_scalar("Loss/val",       val_metrics["loss"],     epoch)
    writer.add_scalar("Accuracy/train", train_acc,               epoch)
    writer.add_scalar("Accuracy/val",   val_metrics["accuracy"], epoch)
    writer.add_scalar("F1/val",         val_metrics["f1"],       epoch)
    writer.add_scalar("AUC/val",        val_metrics["auc"],      epoch)
 
    # Display evaluation metrics on a per-epoch basis
    print(f"Epoch {epoch:02d}/{EPOCHS} | ")
    print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | ")
    print(f"Val Loss: {val_metrics['loss']:.4f}  Val Acc: {val_metrics['accuracy']:.4f}  ")
    print(f"F1: {val_metrics['f1']:.4f}  AUC: {val_metrics['auc']:.4f}")
 
    # Early stopping: save checkpoint if val loss improved, otherwise increment patience counter
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Val loss improved - checkpoint saved to {best_model_path}")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement ({epochs_without_improvement}/{PATIENCE})")
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break
 
writer.close()
 
# Restore best checkpoint before final evaluation on held-out test set
print(f"\nRestoring best model weights from {best_model_path}...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
 
 
# ==================== Final Evaluation & Metric Reporting ====================
 
# Evaluate model on the held-out test set after training is complete
final_metrics, final_preds, final_labels, final_probs = evaluate(
    model, test_loader, criterion, device
)
 
class_names = ['Negative', 'Neutral', 'Positive']
 
print("\n========= Final Test Set Results =========")
print(f"Accuracy  : {final_metrics['accuracy']:.4f}")
print(f"Precision : {final_metrics['precision']:.4f}  (macro)")
print(f"Recall    : {final_metrics['recall']:.4f}  (macro)")
print(f"F1        : {final_metrics['f1']:.4f}  (macro)")
print(f"AUC       : {final_metrics['auc']:.4f}  (macro OvR)")
 
print("\nPer-class report:")
print(classification_report(final_labels, final_preds, target_names=class_names, digits=4))
 
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(final_labels, final_preds))
 
# Log hyperparameters alongside final test metrics for the reproducibility package/ablation study
os.makedirs("results", exist_ok=True)
results_row = {
    "run":          run_name,
    "hidden_dim":   HIDDEN_DIM,
    "num_heads":    NUM_HEADS,
    "dropout":      DROPOUT,
    "epochs":       EPOCHS,
    "lr":           LR,
    "batch_size":   BATCH_SIZE,
    "patience":     PATIENCE,
    "accuracy":     final_metrics["accuracy"],
    "precision":    final_metrics["precision"],
    "recall":       final_metrics["recall"],
    "f1":           final_metrics["f1"],
    "auc":          final_metrics["auc"],
}
results_df = pd.DataFrame([results_row])
results_path = os.path.join("results", "novel_model_results.csv")
write_header = not os.path.exists(results_path)
results_df.to_csv(results_path, mode='a', header=write_header, index=False)
print(f"\nMetrics saved to {results_path}")

# Save prediction arrays for evaluate.py (ROC curves, confusion matrices)
npz_path = os.path.join("results", "novel_model_preds.npz")
np.savez(npz_path, labels=final_labels, preds=final_preds, probs=final_probs)
print(f"Predictions saved to {npz_path}")

 
print("\n========= Training Complete =========")
