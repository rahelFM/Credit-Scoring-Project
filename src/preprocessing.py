import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
# Set file paths
train_path = 'C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv'
test_path = 'C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\test.csv'
# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# Load the RFM data (if you didn't carry it over)
rfm = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\rfm_metrics.csv')

# 1. Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
# Convert TransactionStartTime to datetime
train['TransactionStartTime'] = pd.to_datetime(train['TransactionStartTime'])

# Set snapshot date for Recency
snapshot_date = train['TransactionStartTime'].max() + pd.Timedelta(days=1)

# Calculate RFM
rfm = train.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()

rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
# Scale RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Cluster with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
# Analyze cluster characteristics
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# Assume low Frequency & Monetary = High risk
high_risk_cluster = cluster_summary.sort_values(by=['Frequency', 'Monetary']).index[0]

# Assign is_high_risk label
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# Final customer_risk_df
customer_risk_df = rfm[['CustomerId', 'is_high_risk']]
train = train.merge(customer_risk_df, on='CustomerId', how='left')
print(train['is_high_risk'].value_counts(dropna=False))

# Select the numerical features
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
rfm_scaled = scaler.fit_transform(rfm_features)

# Optional: check the shape or first few rows
print(rfm_scaled[:5])
#clustering with k-means
from sklearn.cluster import KMeans

# Set number of clusters
k = 3

# Initialize KMeans
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit KMeans on scaled RFM features
clusters = kmeans.fit_predict(rfm_scaled)

# Add the cluster labels back to the original rfm_df
rfm['Cluster'] = clusters

print(rfm.describe())

import seaborn as sns
import matplotlib.pyplot as plt

rfm_cols = ['Recency', 'Frequency', 'Monetary']

for col in rfm_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(rfm[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
#checking for missing values
print(rfm.isnull().sum())
#outlier detection with boxplots
for col in rfm_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=rfm[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()



