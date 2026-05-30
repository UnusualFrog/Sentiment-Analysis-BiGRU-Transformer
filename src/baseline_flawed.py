import random

import pandas as pd 
import numpy as np
import torch
import tensorflow as tf

from imblearn.over_sampling import SMOTE


# Read raw dataset
df = pd.read_csv('data/Reviews.csv')

# Display dataset info and sample rows
df.info()
print(df.head(3))

# Drop irrelevant features
df = df.drop(["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time", "Summary"], axis=1)

# Split training features from target feature
X = df.drop("Score", axis=1)
y = df["Score"]

print(X.head(3))
print(y.head(3))

# Apply SMOTE before split (generally incorrect, but accurate replication)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X,y = smote.fit_resample(X,y)

print(X.head(3))
print(y.head(3))