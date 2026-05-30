import pandas as pd 
import numpy as np 

# Set Global Random Seed for reproducability
GLOBAL_SEED = 42;
np.random.seed(GLOBAL_SEED)

# Read raw dataset
df = pd.read_csv('data/Reviews.csv')

# Display dataset info and sample rows
df.info()
print(df.head(3))