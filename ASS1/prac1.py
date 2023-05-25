# Assignment No. 1 - Data Wrangling I:
# Program by SIR, written by your friends ^_^

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('Placement_Data_Full_Class.csv')
print("Placement dataset is successfully loaded into DataFrame")

# Display information of dataset
print("Information of Dataset:\n", df.info())
print("Shape of Dataset (row * column):\n", df.shape)
print("Columns names:\n", df.columns)
print("Number of elements in dataset:", df.size)
print("Datatype or attributes (columns):\n", df.dtypes)

# Display first 5 rows
print("First 5 rows:\n", df.head())

# Display last 5 rows
print("Last 5 rows:\n", df.tail())

# Display any 5 random rows
print("Any 5 rows:\n", df.sample(5))

# Display statistical information of dataset
print("Statistical information of Numerical Columns:\n", df.describe())

# Display null values
print("Total Number of Null Values in Dataset:", df.isna().sum())

# Data type conversion
df['sl_no'] = df['sl_no'].astype('int8')
print("Check Datatype of sl_no:", df.dtypes['sl_no'])
df['ssc_p'] = df['ssc_p'].astype('int8')
print("Check Datatype of ssc_p:", df.dtypes['ssc_p'])

# Label Encoding Conversion of Categorical to Quantitative
df['gender'] = df['gender'].astype('category')
print("Data type of gender:", df.dtypes['gender'])
df['gender'] = df['gender'].cat.codes
print("Data type of gender after label encoding:\n", df['gender'])
print("gender Values:\n", df["gender"].unique())

# Normalization
print("Normalization using Min-Max Feature Scaling:")
df['salary'] = (df['salary'] - df['salary'].min()) / (df['salary'].max() - df['salary'].min())

print(df.head().T)