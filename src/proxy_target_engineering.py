# File: src/target_engineering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Paths
RAW_DATA_PATH = "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv"
FEATURED_DATA_PATH = "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\feature_engineered_data.csv"
OUTPUT_PATH = "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\final_model_data.csv"

def compute_rfm(df):
    # Convert TransactionStartTime to datetime and remove timezone
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (pd.to_datetime('2025-07-01') - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm



def cluster_rfm(rfm_df, n_clusters=3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Optional: sort clusters by risk manually based on domain knowledge
    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_summary['Monetary'].idxmin()  # Low spenders as high risk (customizable)

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

def main():
    print("Loading original transaction data...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    rfm_df = compute_rfm(raw_df)

    print("Clustering customers using RFM...")
    rfm_labeled = cluster_rfm(rfm_df)

    print("Loading feature-engineered dataset...")
    features_df = pd.read_csv(FEATURED_DATA_PATH)

    print("Merging is_high_risk label with features...")
    final_df = features_df.merge(rfm_labeled, on='CustomerId', how='left')

    print("Saving final dataset with proxy target variable...")
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Final dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
