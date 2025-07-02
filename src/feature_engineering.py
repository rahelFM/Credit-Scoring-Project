import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime

class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert to datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')

        # Extract datetime features
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year

        return X

class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        grouped = X.groupby('CustomerId')['Amount'].agg([
            ('TotalTransactionAmount', 'sum'),
            ('AverageTransactionAmount', 'mean'),
            ('TransactionCount', 'count'),
            ('TransactionAmountStd', 'std')
        ]).reset_index()
        return X.merge(grouped, on='CustomerId', how='left')

# Define numerical and categorical columns
numerical_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                      'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'TransactionAmountStd']
categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

# Pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

full_pipeline = Pipeline([
    ('feature_extraction', TransactionFeatureExtractor()),
    ('aggregation', AggregateCustomerFeatures()),
    ('preprocessing', preprocessor)
])

if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv')

    # Apply feature extraction + aggregation only (before ColumnTransformer)
    df_transformed = Pipeline([
        ('feature_extraction', TransactionFeatureExtractor()),
        ('aggregation', AggregateCustomerFeatures())
    ]).fit_transform(df)

    # Run full preprocessing separately
    df_prepared = preprocessor.fit_transform(df_transformed)

    # Get column names from transformers
    cat_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_columns = numerical_features + list(cat_columns)

    # Convert to DataFrame
    df_processed = pd.DataFrame(df_prepared.toarray() if hasattr(df_prepared, 'toarray') else df_prepared,
                                columns=all_columns)

    # Add CustomerId back
    df_processed['CustomerId'] = df_transformed['CustomerId'].values

    # Save
    output_path = 'C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\feature_engineered_data.csv'
    df_processed.to_csv(output_path, index=False)

    print(f"Processed data saved: {output_path}")
    print(df_processed.shape)
