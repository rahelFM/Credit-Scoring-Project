# load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix


# Set file paths
train_path = 'C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv'
test_path = 'C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\test.csv'
# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# Show dimensions
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
# Preview the dataset
print(train.head())

# Check column data types
print(train.dtypes)
# Parse transaction time column
train['TransactionStartTime'] = pd.to_datetime(train['TransactionStartTime'])
test['TransactionStartTime'] = pd.to_datetime(test['TransactionStartTime'])
# Summary stats
print(train.describe(include='all'))

# Set snapshot date (the "present" moment to calculate recency from)
snapshot_date = train['TransactionStartTime'].max() + pd.Timedelta(days=1)

# Only keep positive transactions (i.e., actual spending)
positive_txns = train[train['Amount'] > 0]

# Calculate RFM for each customer
rfm = positive_txns.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()

# Rename columns
rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

# Display RFM head
print(rfm.head())

# Save RFM to processed data folder
rfm.to_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\rfm_metrics.csv', index=False)