# Assiorment no. 10 - data Visualization on Titanic Dataset (histogram and boxplot)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('IRIS1.csv')
print('Boston dataset loaded.')

# Display information about dataset
print('Information of dataset:\n', df.info())
print('Shape of dataset (row x column):', df.shape)
print('Column names:', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)
# Find missing values
print('Missing values:')
print(df.isnull().sum())

# Histogram of 1-variable
fig, axes = plt.subplots(2, 2)
fig.suptitle('Histogram of 1-variables')
sns.histplot(data=df, x='sepal.length', ax=axes[0, 0])
sns.histplot(data=df, x='sepal.width', ax=axes[0, 1])
sns.histplot(data=df, x='petal.length', ax=axes[1, 0])
sns.histplot(data=df, x='petal.width', ax=axes[1, 1])
plt.show()

# Histogram of 2-variables
fig, axes = plt.subplots(2, 2)
fig.suptitle('Histogram of 2-variables')
sns.histplot(data=df, x='sepal.length', multiple='dodge', ax=axes[0, 0])
sns.histplot(data=df, x='sepal.width', multiple='dodge', ax=axes[0, 1])
sns.histplot(data=df, x='petal.length', multiple='dodge', ax=axes[1, 0])
sns.histplot(data=df, x='petal.width',multiple='dodge', ax=axes[1, 1])
plt.show()

# Boxplot of 1-variable
fig, axes = plt.subplots(2, 2)
fig.suptitle('Boxplot of 1-variables')
sns.boxplot(data=df, x='sepal.length', ax=axes[0, 0])
sns.boxplot(data=df, x='sepal.width', ax=axes[0, 1])
sns.boxplot(data=df, x='petal.length', ax=axes[1, 0])
sns.boxplot(data=df, x='petal.width', ax=axes[1, 1])
plt.show()