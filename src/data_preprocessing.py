from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

# 2. Elbow method to find optimal number of clusters
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 10), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for K-means')
plt.show()
# 3. Run KMeans with optimal k (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 4. Analyze cluster profiles
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print(cluster_summary)

# Based on the summary, Cluster 0 is least active → high risk
high_risk_cluster = 0

# Add binary target
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# Check how many are high risk
print(rfm['is_high_risk'].value_counts())
#lets merge the risk level to transaction dataset
# Load transaction-level dataset
transactions = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv')

# Merge with RFM labels
# Merge high-risk labels into the main train dataset

transactions = transactions.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Check result
print(transactions[['CustomerId', 'is_high_risk']].head())
print(transactions['is_high_risk'].value_counts(dropna=False))


#lets see why missing happened
missing_customers = transactions[transactions['is_high_risk'].isna()]['CustomerId'].unique()
print(f"Number of unique customers with missing label: {len(missing_customers)}")
print(missing_customers[:10])  # print first 10 as sample
#number of missing customers are 110, so lets see if they in RFM dataset?
# Assuming 'rfm' is your RFM dataframe with CustomerId and is_high_risk

rfm_customers = rfm['CustomerId'].unique()
missing_in_rfm = [c for c in missing_customers if c not in rfm_customers]
print(f"Number of missing customers not in RFM: {len(missing_in_rfm)}")
print(missing_in_rfm[:10])  # sample
#These customers either had no transactions within the RFM snapshot period I used to calculate Recency, Frequency, Monetary.
#Check if these customers have any transactions in the train dataset at all, and when those transactions happened. Maybe their transactions are outside the snapshot window used for RFM.

missing_customers = ['CustomerId_7476', 'CustomerId_7437', 'CustomerId_7409', 'CustomerId_4552', 'CustomerId_7426', 'CustomerId_7412', 'CustomerId_7343', 'CustomerId_7430', 'CustomerId_7428', 'CustomerId_7450']

missing_transactions = train[train['CustomerId'].isin(missing_customers)]
print(missing_transactions[['CustomerId', 'TransactionStartTime']])
#The CustomerIds with missing is_high_risk labels do have transactions in my data — but they were excluded from the RFM table, which is what the label is based on.
#so lets inspect why they were excluded
print(missing_transactions['Amount'].describe())
print(missing_transactions['Amount'].value_counts().head(10))
print(missing_transactions['CustomerId'].nunique())
#now it is clear that all the CustomerIds have only negative Amount values,In this dataset, it likely indicates refunds or credits — i.e., money going into the customer's account, not spent.
#So these customers:
#Never made a positive purchase (debit),
#Only received refunds or wallet top-ups,
#Didn’t engage in behavior measurable by RFM (Recency, Frequency, Monetary).
#That’s why your RFM aggregation excluded them — because they have:
#Monetary value = 0 (or meaningless negative),
#Frequency = 0 (if filtered),
#Recency not calculated (if no positive reference point).
#So, since they have non-standard behaviour, lets drop them. 
# Drop customers with missing high-risk label
# Merge high-risk labels into the main train dataset