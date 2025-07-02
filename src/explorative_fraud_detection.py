# load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

# stop warnings
import warnings 
warnings.filterwarnings('ignore')
# display column limita
pd.set_option('display.max_columns',500)
# load data
train = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\training.csv')
validation = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\test.csv')
print(train.head())
# checking the balance of the data
print('The number of Non-Frauds are: ' + str(train['FraudResult'].value_counts()[0]) + ' which is', round(train['FraudResult'].value_counts()[0]/len(train) * 100,2), '% of the dataset')
print('The number of Frauds are: ' + str(train['FraudResult'].value_counts()[1]) + ' which is', round(train['FraudResult'].value_counts()[1]/len(train) * 100,2), '% of the dataset')
sns.countplot(x='FraudResult', data=train)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()
#SMOTE
# SMOTE
# oversampling
from imblearn.over_sampling import SMOTE

count_class_0, count_class_1 = train.FraudResult.value_counts()

# divide by class
train_class_0 = train[train['FraudResult'] == 0]
train_class_1 = train[train['FraudResult'] == 1]
train_class_1_over = train_class_1.sample(count_class_0, replace=True)
train_test_over = pd.concat([train_class_0, train_class_1_over], axis=0)

print('Random over-sampling:')
print(train_test_over.FraudResult.value_counts())

train_test_over.FraudResult.value_counts().plot(kind='bar', title='Count (FraudResult)');
plt.show()
#new training after oversampling
train1 = train_test_over 
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns
# pricing and fraudresults
sns.countplot(y='ProviderId', data=train1, hue='FraudResult')
plt.show
# pricingstrategy and fraudresult
sns.countplot(x='PricingStrategy', data=train1, hue='FraudResult')
plt.show()
# product category and fraudresult
sns.countplot(y='ProductCategory',data = train1, hue = 'FraudResult')
# ProductId and fraudresult
sns.countplot(y='ProductId', data = train1, hue = 'FraudResult')
# channelid and fraudresult
sns.countplot(x='ChannelId', data = train1, hue = 'FraudResult')
